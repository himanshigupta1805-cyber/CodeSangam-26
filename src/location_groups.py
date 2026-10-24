"""Build shared theater entities from exact latitude/longitude pairs."""

import pandas as pd


def build_location_groups(
    booknow_theaters,
    cinepos_theaters,
    booknow_ids=None,
    cinepos_ids=None,
    theater_relation=None,
):
    """Return one row per source theater ID with a shared location group.

    Exact coordinate pairs identify a shared theater across both source
    systems. Missing coordinates do not imply a match. A supplied crosswalk
    joins the full groups of its paired IDs, even when recorded coordinates
    differ.
    """
    source_tables = (
        ("booknow", booknow_theaters, "book_theater_id", booknow_ids),
        ("cinepos", cinepos_theaters, "cine_theater_id", cinepos_ids),
    )
    frames = []
    for source, theaters, id_column, all_ids in source_tables:
        frame = theaters[[id_column, "latitude", "longitude"]].copy()
        frame = frame.dropna(subset=[id_column]).drop_duplicates(id_column)
        frame = frame.rename(columns={id_column: "theater_id"})
        if all_ids is not None:
            known_ids = set(frame["theater_id"])
            additional_ids = sorted(
                set(all_ids).difference(known_ids), key=str
            )
            if additional_ids:
                frame = pd.concat(
                    [
                        frame,
                        pd.DataFrame(
                            {
                                "theater_id": additional_ids,
                                "latitude": float("nan"),
                                "longitude": float("nan"),
                            }
                        ),
                    ],
                    ignore_index=True,
                )
        frame["source"] = source
        frames.append(frame)

    entities = pd.concat(frames, ignore_index=True)
    has_coordinates = entities[["latitude", "longitude"]].notna().all(axis=1)

    # Factorize coordinate tuples across both systems so equal pairs share an ID.
    coordinate_keys = list(
        zip(entities.loc[has_coordinates, "latitude"],
            entities.loc[has_coordinates, "longitude"])
    )
    codes, _ = pd.factorize(pd.Series(coordinate_keys, dtype="object"), sort=True)
    entities["location_group_id"] = pd.Series(index=entities.index, dtype="object")
    entities.loc[has_coordinates, "location_group_id"] = [
        f"location_{code:04d}" for code in codes
    ]

    missing_rows = entities.loc[~has_coordinates]
    entities.loc[~has_coordinates, "location_group_id"] = [
        f"unlocated_{source}_{theater_id}"
        for source, theater_id in zip(
            missing_rows["source"], missing_rows["theater_id"]
        )
    ]

    # A supplied crosswalk joins the two full groups, propagating the link to
    # every same-coordinate ID on either side. The crosswalk also bridges
    # coordinate groups whose recorded coordinates differ.
    if theater_relation is not None:
        row_lookup = {
            (row.source, row.theater_id): index
            for index, row in entities.iterrows()
        }
        for relation in theater_relation.drop_duplicates().itertuples(index=False):
            book_index = row_lookup.get(("booknow", relation.book_theater_id))
            cine_index = row_lookup.get(("cinepos", relation.cine_theater_id))
            if book_index is None or cine_index is None:
                continue
            book_has_coords = has_coordinates.loc[book_index]
            cine_has_coords = has_coordinates.loc[cine_index]
            book_group = entities.at[book_index, "location_group_id"]
            cine_group = entities.at[cine_index, "location_group_id"]
            if book_group == cine_group:
                continue

            if book_has_coords:
                shared_group = book_group
            elif cine_has_coords:
                shared_group = cine_group
            else:
                shared_group = f"relation_{relation.book_theater_id}_{relation.cine_theater_id}"
            entities.loc[
                entities["location_group_id"].isin([book_group, cine_group]),
                "location_group_id",
            ] = shared_group

    return entities[
        ["source", "theater_id", "latitude", "longitude", "location_group_id"]
    ]
