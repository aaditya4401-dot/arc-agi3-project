"""Conservative source-only function retention; no snippet state is serialized.

Also embedded into the isolated Python runner. Keep this module stdlib-only.
"""
import ast
import dis
import sys
import types


def _import_statements(nodes):
    """One explicit absolute import per binding; never replay wildcard imports."""
    statements = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                binding = alias.asname or alias.name.split('.')[0]
                statements[binding] = ast.unparse(ast.Import(names=[alias]))
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            for alias in node.names:
                if alias.name == '*':
                    raise ValueError('wildcard imports are not retained; use explicit imports inside the function')
                statements[alias.asname or alias.name] = ast.unparse(
                    ast.ImportFrom(module=node.module, names=[alias], level=0))
        else:
            raise ValueError('only explicit absolute imports can accompany retained functions')
    return statements


def _source_parts(source, retain_imports=True):
    tree = ast.parse(source, '<retained>', 'exec')
    if (not tree.body or not isinstance(tree.body[-1], ast.FunctionDef)
            or (not retain_imports and len(tree.body) != 1)):
        raise ValueError('only top-level function definitions can be retained')
    imports = _import_statements(tree.body[:-1])
    return tree.body[-1], imports


def function_signature(source):
    node, _ = _source_parts(source)
    return node.name + '(' + ast.unparse(node.args) + ')'


def _function_record(source, retain_imports=False):
    node, imports = _source_parts(source, retain_imports)
    if getattr(node, "type_params", []):
        raise ValueError("generic type parameters are not retained")
    if node.decorator_list:
        raise ValueError('decorated functions are not retained')
    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    args += [a for a in (node.args.vararg, node.args.kwarg) if a is not None]
    if node.returns is not None or any(a.annotation is not None for a in args):
        raise ValueError('use an unannotated definition for retention')
    # Restoration must never evaluate model calls or snapshot a frame via defaults.
    for default in [*node.args.defaults, *node.args.kw_defaults]:
        if default is not None:
            try:
                ast.literal_eval(default)
            except Exception:
                raise ValueError('defaults must be literals; pass changing data as arguments') from None
    tree = ast.Module(body=[node], type_ignores=[])
    code = next(c for c in compile(tree, '<retained>', 'exec').co_consts
                if isinstance(c, types.CodeType))
    reads = set()
    pending = [code]
    while pending:
        current = pending.pop()
        for instruction in dis.get_instructions(current):
            if instruction.opname in ('LOAD_GLOBAL', 'LOAD_NAME'):
                reads.add(instruction.argval)
            elif instruction.opname in ('STORE_GLOBAL', 'DELETE_GLOBAL'):
                raise ValueError('functions that assign global variables are not retained')
        pending.extend(c for c in current.co_consts if isinstance(c, types.CodeType))
    return node.name, source, reads - imports.keys()


def _rejection(name, reason, hint=None, **metadata):
    result = {'name': name, 'reason': reason, **metadata}
    if hint:
        result['hint'] = hint
    return result


def _validation_hint(reason):
    if 'decorat' in reason:
        return 'Use a plain top-level def without decorators if you want to reuse it.'
    if 'annotat' in reason:
        return 'Remove parameter and return annotations if you want to retain this function.'
    if 'default' in reason:
        return 'Use literal defaults and pass changing values explicitly as arguments.'
    if 'global variables' in reason:
        return 'Return updated values instead of assigning global variables.'
    if 'import' in reason:
        return 'Put explicit, sandbox-supported imports inside the function.'
    return 'Use a plain top-level def if you want to retain this function.'


def prepare_functions(sources, available, *, retain_imports=False):
    """Validate sources without executing them; drop invalid dependency chains."""
    records, rejected = {}, []
    for source in sources:
        name = '<unknown>'
        try:
            parsed = ast.parse(source)
            if (parsed.body and isinstance(parsed.body[-1], ast.FunctionDef)
                    and (retain_imports or len(parsed.body) == 1)):
                name = parsed.body[-1].name
                records.pop(name, None)
            name, source, reads = _function_record(source, retain_imports)
            _, imports = _source_parts(source)
            if imports.keys() & set(available):
                raise ValueError('import alias conflicts with a provided global or builtin')
            records[name] = (source, reads)
        except Exception as exc:
            reason = str(exc)[:180]
            rejected.append(_rejection(name, reason, _validation_hint(reason)))
    # Import bundles must not overwrite another retained function at restore.
    function_names = set(records)
    for name, (source, _) in list(records.items()):
        _, imports = _source_parts(source)
        if imports.keys() & function_names:
            del records[name]
            rejected.append(_rejection(name, 'import alias conflicts with a retained function',
                                       'Choose a different import alias or function name.'))
    # A wrapper must not survive its unavailable dependency. Re-run to a fixed point.
    while True:
        dropped = []
        for name, (source, reads) in records.items():
            missing = reads - set(available) - records.keys()
            if missing:
                dropped.append(name)
                names = ', '.join(sorted(missing))
                rejected.append(_rejection(
                    name, 'unavailable dependencies: ' + names,
                    'Pass ' + names + ' as arguments, or put their imports inside the function.',
                    missing_dependencies=sorted(missing)))
        if not dropped:
            break
        for name in dropped:
            del records[name]
    return {name: value[0] for name, value in records.items()}, rejected


def restore_functions(sources, namespace, available, *, retain_imports=False,
                      repair_hints=False):
    """Restore definitions and explicit imports through the sandbox importer.

    Never replay assignments, expressions, decorators, or previous actions.
    """
    kept, rejected = prepare_functions(sources, available, retain_imports=retain_imports)
    failed = set()
    for name, source in kept.items():
        before = dict(namespace)
        try:
            if name in available:
                raise ValueError('name conflicts with a provided global or builtin')
            exec(compile(source, '<retained>', 'exec'), namespace, namespace)
        except Exception:
            # A multi-import bundle can fail partway through. Restore bindings
            # so one failed helper does not replace another helper's globals.
            namespace.clear()
            namespace.update(before)
            failed.add(name)
            reason = ('could not restore this definition or its imports' if repair_hints else
                      'could not restore this definition; redefine it if needed')
            rejected.append(_rejection(name, reason,
                                       'Check its imports are supported by the sandbox, then redefine it if needed.'))
    survivors, dependencies = prepare_functions(
        [source for name, source in kept.items() if name not in failed], available,
        retain_imports=retain_imports)
    rejected.extend(dependencies)
    for name in kept.keys() - survivors.keys():
        if name not in available:
            namespace.pop(name, None)
    return survivors, rejected


def _matching_imports(nodes, namespace):
    """Recognize live import bindings without importing or evaluating anything.

    Only module bindings and callable/immutable exports are eligible. Ordinary
    snippet assignments and mutable imported data are never serialized. Looking
    in module dictionaries avoids invoking __getattr__ during bookkeeping.
    """
    matches, declared = {}, {}
    for node in nodes:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        try:
            statements = _import_statements([node])
        except ValueError:
            continue
        declared.update(statements)
        for binding, statement in statements.items():
            single = ast.parse(statement).body[0]
            alias = single.names[0]
            if isinstance(single, ast.Import):
                # Even unaliased dotted imports must already have executed.
                if alias.name not in sys.modules:
                    continue
                module_name = alias.name if alias.asname else alias.name.split('.')[0]
                expected = sys.modules.get(module_name)
            else:
                module = sys.modules.get(single.module)
                expected = vars(module).get(alias.name) if type(module) is types.ModuleType else None
            eligible = type(expected) in (
                types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType,
                type, str, bytes, int, float, complex, bool,
            )
            if eligible and binding in namespace and namespace[binding] is expected:
                matches[binding] = statement
    return matches, declared


def collect_functions(code_text, previous, namespace, available, *, retain_imports=False,
                      repair_hints=False):
    """Collect from the actual namespace, including after partial execution.

    A later def may never have run. Match the live function's source location
    rather than choosing the last definition in the submitted snippet.
    """
    candidates = {}
    rejected = []
    try:
        tree = ast.parse(code_text, '<python_tool>')
    except SyntaxError:
        # Parsing failed before execution; restored functions are still valid.
        tree = ast.Module(body=[], type_ignores=[])
    fresh = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fresh.setdefault(node.name, []).append(node)
    import_nodes = []
    for source in previous.values():
        # Old source-only records remain valid; no migration is necessary.
        import_nodes.extend(ast.parse(source).body[:-1])
    import_nodes.extend(tree.body)
    imports, declared_imports = (_matching_imports(import_nodes, namespace)
                                if retain_imports or repair_hints else ({}, {}))
    for name in dict.fromkeys([*previous, *fresh]):
        value = namespace.get(name)
        if not repair_hints and (name in available or type(value) is not types.FunctionType
                                 or value.__name__ != name):
            rejected.append(_rejection(name, 'definition did not execute, was removed/replaced, or conflicts with a provided name'))
            continue
        if name in available:
            rejected.append(_rejection(name, 'name conflicts with a provided global or builtin',
                                       'Choose a different function name.'))
            continue
        if name not in namespace:
            reason = 'function was removed' if name in previous else 'definition was not reached or was removed'
            rejected.append(_rejection(name, reason, 'Define it before calling it if it is still needed.'))
            continue
        if type(value) is not types.FunctionType or value.__name__ != name:
            rejected.append(_rejection(name, 'function name was replaced by another value',
                                       'Use a separate variable name; redefine the function only if needed.'))
            continue
        if name in previous and value.__code__.co_filename == '<retained>':
            # Preserve the restored definition when execution stopped before a
            # replacement (or while evaluating that replacement's defaults).
            candidates[name] = previous[name]
            continue
        node = next((node for node in fresh.get(name, [])
                     if isinstance(node, ast.FunctionDef) and not node.decorator_list
                     and value.__code__.co_filename == '<python_tool>'
                     and value.__code__.co_firstlineno == node.lineno), None)
        if node is None:
            rejected.append(_rejection(name, 'definition was decorated, replaced outside a top-level def, or is unsupported',
                                       'Use a plain top-level def without decorators if you want to reuse it.'))
            continue
        candidates[name] = ast.get_source_segment(code_text, node)
    # Rebuild every bundle from the LIVE namespace. An old helper must not
    # silently restore an obsolete import when that alias was rebound/deleted.
    for name, source in (list(candidates.items()) if retain_imports else []):
        try:
            node, _ = _source_parts(source)
            definition = ast.get_source_segment(source, node)
            _, _, reads = _function_record(definition)
            statements = [imports[n] for n in sorted(reads & imports.keys()) if n not in available]
            candidates[name] = '\n'.join([*statements, definition])
        except Exception:
            # Validation below produces the specific rejection and prunes callers.
            pass
    kept, invalid = prepare_functions(candidates.values(), available, retain_imports=retain_imports)
    for item in invalid:
        missing = item.get('missing_dependencies', [])
        if not missing:
            continue
        hints = []
        missing_imports = [n for n in missing if n in declared_imports]
        missing_helpers = [n for n in missing if n in candidates and n not in declared_imports]
        missing_data = sorted(set(missing) - set(missing_imports) - set(missing_helpers))
        if missing_imports:
            statements = '; '.join(declared_imports[n] for n in missing_imports)
            hints.append('Put `' + statements + '` inside this function if you want to reuse it.')
        if missing_helpers:
            hints.append('Make the helper dependencies retainable too: ' + ', '.join(missing_helpers) + '.')
        if missing_data:
            hints.append('Pass ' + ', '.join(missing_data) + ' as arguments; snippet variables are not retained.')
        item['hint'] = ' '.join(hints)
    return kept, rejected + invalid
