import clang.cindex
import jinja2
# import json
# import pprint
import inspect

# Initialize Clang
clang.cindex.Index.create()

# Parse the header file
index = clang.cindex.Index.create()
target_location = "../../src/llama-model.h"
# target_location = "../../src/llama-model.cpp"
translation_unit = index.parse(target_location)

# Traverse the AST to collect function declarations
declarations = []
functions = []
structs = []
enums = []
classes = []
constants = []

template = jinja2.Template("""#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(llama_model, m) {
{%- for declaration in declarations %}
    {{ declaration }}
{% endfor -%}
}
""")

function_template = jinja2.Template("""m.def("{{ func.name }}", &{{ func.name }}{% if func.desc -%}, "{{ func.desc }}{%- endif %}");""")

struct_template = jinja2.Template("""py::class_<{{ struct.name }}>(m, "{{ struct.name }}")
        .def(py::init<>())
{%- for param in struct.parameters %}
        .def_readwrite("{{ param.name }}", &{{ struct.name }}::{{ param.name }})
{%- endfor %};""")

constant_template = jinja2.Template("""py::bind_constant(m, "{{ constant.name }}", &{{ constant.name }});""")

enum_template = jinja2.Template("""py::enum_<{{ enum.type }}>(m, "{{ enum.name }}")
{%- for value in values %}
        .value("{{ value.name }}", {{ enum.type }}::{{ value.name }})
{%- endfor %};""")


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
            elif inspect.isclass(value):
                print(f"{key}: {value}")
                dump(value)
            print(f"{key}: {value}")
        except (Exception) as e:
            print(e, key)


def visit_node(node):
    if str(node.location.file) == target_location:
        # Extract function name, return type, parameters
        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
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
        # Handle Structs        
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
            # Don't include re-exports
            if len(parameters) > 0:
                structs.append({
                    "name": name,
                    "parameters": parameters,
                })
                declarations.append(struct_template.render(struct=structs[-1]))
        # Handle classes
        elif node.kind == clang.cindex.CursorKind.CLASS_DECL:
            pass
        # Handle Enums
        elif node.kind == clang.cindex.CursorKind.ENUM_DECL:
            # Get enum name
            enum_name = node.spelling

            # Determine namespace (if any)
            namespace = ""
            parent = node.semantic_parent
            while parent:
                if parent.kind == clang.cindex.CursorKind.NAMESPACE:
                    namespace = parent.spelling
                    break
                parent = parent.semantic_parent

            # Fully qualified enum type (e.g., "MyNS::Color")
            enum_type = f"{namespace}::{enum_name}" if namespace else enum_name

            # Collect enum values (ENUM_CONSTANT_DECL)
            values = list()
            for child in node.get_children():
                if child.kind == clang.cindex.CursorKind.ENUM_CONSTANT_DECL:
                    values.append({"name": child.spelling})
            
            enums.append({
                "name": enum_name,
                "namespace": namespace,
                "type": enum_type,
                "values": values
            })
            declarations.append(enum_template.render(enum=enums[-1], values=enums[-1]["values"]))
        # Handle constants
        elif node.kind == clang.cindex.CursorKind.VAR_DECL and node.spelling.isupper() and node.semantic_parent.spelling == target_location:
            name = node.spelling
            constants.append({
                "name": name,
            })
            declarations.append(constant_template.render(constant=constants[-1]))
        else:
            # if str(node.kind).endswith("_DECL"):
            #    print(str(node.kind))
            pass
    for child in node.get_children():
        visit_node(child)


visit_node(translation_unit.cursor)

# Print pybind11 file
print(template.render(declarations=declarations))
