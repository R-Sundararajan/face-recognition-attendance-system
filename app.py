"""Local date-filtered attendance viewer."""

import sqlite3

from flask import Flask, render_template, request

from attendance_core import DATABASE, attendance_for_date


def create_app(database=DATABASE):
    app = Flask(__name__)
    app.config["DATABASE"] = database

    @app.get("/")
    def index():
        return render_template("index.html", selected_date="", attendance_data=[])

    @app.post("/attendance")
    def attendance():
        selected_date = request.form.get("selected_date", "")
        try:
            rows = attendance_for_date(selected_date, app.config["DATABASE"])
        except ValueError as error:
            return render_template("index.html", selected_date="", attendance_data=[], error=str(error)), 400
        except sqlite3.Error:
            app.logger.exception("Unable to read attendance database")
            return render_template(
                "index.html", selected_date=selected_date, attendance_data=[],
                error="Attendance storage is unavailable. Check file permissions and the database schema."
            ), 503
        return render_template("index.html", selected_date=selected_date, attendance_data=rows, no_data=not rows)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=False)
