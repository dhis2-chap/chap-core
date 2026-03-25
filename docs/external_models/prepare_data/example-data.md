# Run models in Chap with example data

## Create a working directory

Create a dedicated folder for this example and navigate into it:

```console
$ mkdir laos-workshop && cd laos-workshop
```

## Download the example dataset

Download the Laos dengue dataset (monthly, admin-1 level) into your working directory:

```console
$ curl -sL -o chap_LAO_admin1_monthly.csv \
    "https://raw.githubusercontent.com/dhis2/climate-health-data/main/lao/chap_LAO_admin1_monthly.csv"
```

The CSV contains ~2800 rows covering 18 provinces from 1998-2010 with columns: `time_period`, `location`, `disease_cases`, `population`, `location_name`, `rainfall`, `mean_temperature`, `mean_relative_humidity`.

---

<div style="margin-bottom: 10rem;" markdown>

Next: [Validate your data](index.md#3-validating-your-data)

</div>
