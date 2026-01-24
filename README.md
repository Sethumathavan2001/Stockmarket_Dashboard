# Sethu Market Dashboard (Django)

Quick setup

1. Create virtualenv and install requirements:
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt

2. Run migrations and start dev server:
   python manage.py migrate
   python manage.py createsuperuser
   python manage.py runserver

3. Admin UI: http://127.0.0.1:8000/admin/ — add Companies, then use the pages:
   - /          (home - sector/industry stats)
   - /daily/    (select a company to view daily data)
   - /5min/     (select a company to view 5-minute data)
   - /live/     (placeholder for live data)
   - /analysis/ (placeholder for analysis)

4. Run management commands (placeholders):
   python manage.py run_eod_daily
   python manage.py run_eod_5min

Next steps
- Implement data fetchers and storage logic for EOD runs
- Add tests and fixtures
- Add charts and real-time data feeds (websockets)
