import pandas as pd
from market.models import Company

file_path = r"D:\Stocks Analysis\New folder\All_Companies.xlsx"

df = pd.read_excel(file_path)
deleted_count, _ = Company.objects.all().delete()
print(f"Deleted {deleted_count} existing companies")
created, updated = 0, 0

for _, row in df.iterrows():
    company, is_created = Company.objects.update_or_create(
        code=row["symbol"],
        defaults={
            "name": row["longname"],
            "exchange": row.get("exchDisp", ""),
            "sector": row.get("sectorDisp", row.get("sector", "")),
            "industry": row.get("industry", ""),
        }
    )

    if is_created:
        created += 1
    else:
        updated += 1

print(f"Import completed: {created} created, {updated} updated")
