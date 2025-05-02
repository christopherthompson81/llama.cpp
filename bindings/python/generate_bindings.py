import clang.cindex

# Initialize Clang
clang.cindex.Index.create()

# Parse the header file
index = clang.cindex.Index.create()
translation_unit = index.parse("../../ggml/include/ggml.h")

# Traverse the AST to collect function declarations
functions = []


def visit_node(node):
    if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
        # Extract function name, return type, parameters
        name = node.spelling
        return_type = node.result_type.spelling
        parameters = []
        for param in node.get_children():
            if param.kind == clang.cindex.CursorKind.PARM_DECL:
                param_name = param.spelling
                param_type = param.type.spelling
                parameters.append((param_name, param_type))
        functions.append({
            "name": name,
            "return_type": return_type,
            "parameters": parameters
        })
    for child in node.get_children():
        visit_node(child)


visit_node(translation_unit.cursor)

# Print collected functions
for func in functions:
    print(f"Function: {func['name']}")
    print(f"  Return Type: {func['return_type']}")
    print("  Parameters:")
    for name, type in func["parameters"]:
        print(f"    {name}: {type}")
    print()
