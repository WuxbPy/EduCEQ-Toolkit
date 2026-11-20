# The crime panel of Section 5.3

Source: data.police.uk street-level crime, published monthly by the UK Home
Office under the Open Government Licence v3.0. The archive used here is the
four force areas covered by our earlier spatio-temporal study, 2023-01 to
2025-12, 4,167,871 records with longitude, latitude, crime type and month.

Section 5.3 uses one slice of it:

    force area   Cambridgeshire Constabulary
    period       2025 (period_id 24 to 35, where period_id = (year-2023)*12 + month-1)
    records      81,959
    categories   14, every published type kept as its own category,
                 including "Anti-social behaviour"
    coordinates  longitude and latitude to six decimals, projected locally with
                 x = (lon - lon0) * 111.320 * cos(lat0), y = (lat - lat0) * 110.574,
                 which is accurate to better than 0.1 per cent over the extent of
                 one county

The published coordinates are snapped by the publisher: 81,959 offences fall on
10,152 distinct locations. Section 5.3 reports this and states what it implies
for the scale at which the panel is decidable.

The archive is not redistributed. `prepare_crime.py` rebuilds the npz from the
monthly CSV files (fields coords, types, time_periods, region_id); the script is
not included because the download is 580 MB of CSV, but the filter above is the
whole of it, and the counts in Section 5.3 are the check that it was applied.
