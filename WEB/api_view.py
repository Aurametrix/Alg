def api_view(view):
    @functools.wraps(view)
    def django_view(request, *args, **kwargs):
        params = {
            param_name: param.annotation(extract(request, param))
            for param_name, param in inspect.signature(view).parameters.items()
        }
        data = view(**params)
        return JsonResponse(asdict(data))
    
    return django_view
