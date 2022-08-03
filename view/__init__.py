

def init_views(app):
    from .api import bp_api
    from .page import bp_page

    app.register_blueprint(bp_api)
    app.register_blueprint(bp_page)
