def build(config):
    
    if type(config) is not dict or "class" not in config:
        return config
    params = {}
    for key in config:
        if key == "class":
            cls = config[key]
        else:
            params[key] = build(config[key])
    return cls(**params)