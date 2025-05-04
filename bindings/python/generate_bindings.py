import clang.cindex
import jinja2
# import pprint
import inspect

# Initialize Clang
clang.cindex.Index.create()

# Parse the header file
index = clang.cindex.Index.create()
# target_location = "../../src/llama-model.cpp"
target_location = "../../src/llama-model.cpp"
translation_unit = index.parse(target_location)

# Traverse the AST to collect function declarations
declarations = []
functions = []
structs = []
enums = []
classes = []
constants = []

template = jinja2.Template("""
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(example, m) {
    {% for declaration in declarations %}{{ declaration }}
    {% endfor %}
}
""")
# template.render(declarations=declarations)

function_template = jinja2.Template("""m.def("{{ func.name }}", &{{ func.name }}, "{{ func.desc }}");""")
# function_template.render(func=func)

struct_template = jinja2.Template("""py::class_<{{ struct.name }}>(m, "{{ struct.name }}")
    .def(py::init<{% for param in struct.parameters %}{{ param.type }}{% if not loop.last %}, {% endif %}{% endfor %}>())
    {% for param in struct.parameters %}.def_readwrite("{{ param.name }}", &{{ struct.name }}::{{ param.name }})
    {% endfor %}""")


def dump(obj):
    for key in dir(obj):
        if key.startswith("__"):
            continue
        try:
            value = getattr(obj, key)
            if inspect.isfunction(value):
                print(f"{key}: {value}")
                dump(value)
            elif inspect.ismethod(value):
                print(f"{key}: {value}")
                dump(value)
            if inspect.isclass(value):
                print(f"{key}: {value}")
                dump(value)
            print(f"{key}: {value}")
        except (Exception) as e:
            print(e, key)


def visit_node(node):
    if str(node.location.file) == target_location:
        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            # Extract function name, return type, parameters
            name = node.spelling
            return_type = node.result_type.spelling
            description = node.brief_comment if node.brief_comment else ""
            parameters = []
            for param in node.get_children():
                if param.kind == clang.cindex.CursorKind.PARM_DECL:
                    param_name = param.spelling
                    param_type = param.type.spelling
                    parameters.append((param_name, param_type))
            functions.append({
                "name": name,
                "desc": description,
                "return_type": return_type,
                "parameters": parameters,
            })
            declarations.append(function_template.render(func=functions[-1]))
        elif node.kind == clang.cindex.CursorKind.STRUCT_DECL:
            name = node.spelling
            parameters = []
            for param in node.get_children():
                if param.kind == clang.cindex.CursorKind.FIELD_DECL:
                    if param.spelling == "":
                        continue
                    param_name = param.spelling
                    param_type = param.type.spelling
                    parameters.append({"name": param_name, "type": param_type})
            structs.append({
                "name": name,
                "parameters": parameters,
            })
            declarations.append(struct_template.render(struct=structs[-1]))
        elif node.kind == clang.cindex.CursorKind.CLASS_DECL:
            pass
        elif node.kind == clang.cindex.CursorKind.ENUM_DECL:
            name = node.spelling
            enums.append({
                "name": name,
            })
        elif node.kind == clang.cindex.CursorKind.VAR_DECL and node.spelling.isupper() and node.semantic_parent.spelling == target_location:
            name = node.spelling
            return_type = node.result_type.spelling
            parameters = []
            for param in node.get_children():
                if param.kind == clang.cindex.CursorKind.PARM_DECL:
                    param_name = param.spelling
                    param_type = param.type.spelling
                    parameters.append((param_name, param_type))
            constants.append({
                "name": name,
                "return_type": return_type,
                "parameters": parameters,
            })
        else:
            # if str(node.kind).endswith("_DECL"):
            #    print(str(node.kind))
            pass
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

# Print collected enums
for struct in structs:
    print(f"Structure: {struct['name']}")
    print()

# Print collected enums
for enum in enums:
    print(f"Enumeration: {enum['name']}")
    print()

# Print collected constants
for const in constants:
    print(f"Constant: {const['name']}")
    print(f"  Return Type: {const['return_type']}")
    print("  Parameters:")
    for name, type in const["parameters"]:
        print(f"    {name}: {type}")
    print()

# Print pybind11 file
print(template.render(declarations=declarations))
