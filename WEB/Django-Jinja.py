import jinja2
from django.template.loader import render_to_string
from django_jinja import library
from django.utils.safestring import mark_safe


@jinja2.contextfunction
@library.global_function
def component(global_ctx, template, **context):
    """
    Render a Jinja2 component.
    Usage:
    {{ component("my-template.html", title="Foo", modifier="bar") }}
    """
    full_context = dict(global_ctx)
    full_context.update(context)
    return mark_safe(render_to_string(template, full_context))
