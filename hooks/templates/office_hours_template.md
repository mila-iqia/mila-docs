Office Hours take place in {{ office_hours_data["location"] }} and on Google Meet:
{% for date in office_hours_data["dates"] %}
* on {{ date.day }}s, from {{ date.start }} to {{ date.end }}
{% endfor %}