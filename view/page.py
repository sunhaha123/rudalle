from flask import Blueprint, render_template


bp_page = Blueprint('bp_page', __name__)


@bp_page.get("/")
def index():
    return render_template("index.html")
