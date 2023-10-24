from flask import Flask, render_template, request
import psycopg2

app = Flask(__name__)


def build_connection_with_database():
    conn = psycopg2.connect(database="cyberbul", host="localhost", port="5432", user="postgres", password="123")
    return conn


def close_connection_with_database(cur, conn):
    conn.commit()
    cur.close()
    conn.close()


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        # print(name, email)
        conn = build_connection_with_database()
        cur = conn.cursor()
        query = f"INSERT INTO users(name, email) VALUES ('{name}', '{email}')"
        cur.execute(query)
        close_connection_with_database(cur, conn)
        return "done Registration"
    return render_template("home.html")


#


if __name__ == "__main__":
    app.run(debug=True)
