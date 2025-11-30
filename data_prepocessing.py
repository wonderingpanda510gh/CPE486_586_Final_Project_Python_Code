import polars as pl
import polars.selectors as cs


def read_and_clean_data(sample_n=10000):
    # import dataset
    print("Import the dataset")

    md = pl.read_excel('dataset/data_movies_clean.xlsx')

    # select the columns needed for the analysis
    md = md.select([
        pl.col("production_company_names"),
        pl.col("origin_country"),
        pl.col("original_language"),
        pl.col("genre_names"),
        pl.col("runtime"),  
        pl.col("vote_average"),
        pl.col("popularity"),
    ])

    # clean the dataset, drop the rows with missing values
    # first, replace any empty lists with nulls and drop them
    # define the values in the dataset that should be considered as nulls
    null_value = ["", "[]", "['']", "null", "None"]
    md = (
        md
        .with_columns(
        cs.string().str.strip_chars()
        )
        .with_columns(
            # replace specified null values with actual nulls
            cs.string().replace(null_value, None)
        )
        .drop_nulls() # drop the rows with null values
    )

    if sample_n and md.height > sample_n:
        # shuffle=True keep the randomness seed = 42 for repurducibility
        md = md.sample(n=sample_n, shuffle=True, seed=42)

    # convert the columns with list-like strings to actual lists
    list_cols = ["production_company_names", "origin_country", "genre_names"]
    
    convert_exprs = []
    for col in list_cols:
        if col in md.columns and md.schema[col] == pl.String:
            convert_exprs.append(
                pl.col(col)
                .str.strip_chars("[]")
                .str.replace_all("'", "")
                .str.split(", ")
            )
    
    if convert_exprs:
        md = md.with_columns(convert_exprs)

    md = md.with_row_index("movie_id") # add a row index for each movie
    print("Dataset Cleaned")

    return md

def linear_mini_hot_encoding(md):

    print("Doing hot encoding for traditional liear regression models")
    # # convert the columns with list-like strings to actual lists
    # list_columns = ["production_company_names", "origin_country", "genre_names"]
    # md_1 = md.with_columns(
    #     [pl.col(col).str.strip_chars("[]").str.replace_all("'","").str.split(", ") for col in list_columns]
    # )

    # deal with the multi label columns
    production_hot = (md.select(["movie_id", "production_company_names"])
                      .explode("production_company_names")
                      .group_by("movie_id")
                      .sum()
                      .select(pl.col("movie_id"), pl.all().exclude("movie_id").name.prefix("company_")))
    country_hot = (md.select(["movie_id", "origin_country"])
                   .explode("origin_country")
                   .group_by("movie_id")
                   .sum()
                   .select(pl.col("movie_id"), pl.all().exclude("movie_id").name.prefix("country_")))
    genre_hot = (md.select(["movie_id", "genre_names"])
                 .explode("genre_names")
                 .group_by("movie_id")
                 .sum()
                 .select(pl.col("movie_id"), pl.all().exclude("movie_id").name.prefix("genre_")))

    # deal with the single label colums
    language_hot = (md.select(["movie_id", "original_language"])
                    .to_dummies("original_language")
                    .select(pl.col("movie_id"), pl.all().exclude("movie_id").name.prefix("lang_")))

    feat_cols = cs.all().exclude("movie_id")
    # combine all the data together
    md_linear = (
        md.select(["movie_id", "runtime", "vote_average", "popularity"])
        .join(production_hot, on="movie_id", how="left")
        .join(country_hot, on="movie_id", how="left")
        .join(genre_hot, on="movie_id", how="left")
        .join(language_hot, on="movie_id", how="left")
        .with_columns(feat_cols.cast(pl.Float64))
        .fill_null(0)  # fill nulls with 0, prevent null values in the dataset
        .fill_nan(0)  # fill NaNs with 0, prevent NaN values in the dataset
        .drop("movie_id")  # drop the movie_id column as it's no longer needed
    )

    print("Hot encoding finished")

    return md_linear

def vocabularies(md, column_name):
    print("Doing encoding for neural network models")
    # create the vocabulary
    vocabularies = (
        md.select(pl.col(column_name).explode())
        .unique()
        .sort(column_name)
        .with_row_index(name=f"{column_name}_id") #generate the id for the vocabulary
    )

    # map the id to the original data
    md_indexed = (
        md.select(["movie_id", column_name])
        .explode(column_name)
        .join(vocabularies, on=column_name, how="left")
        .group_by("movie_id")
        .agg(pl.col(f"{column_name}_id")) # reunion the id to the list
        .sort("movie_id")
    )

    return md_indexed.drop("movie_id"), len(vocabularies) + 1

def neural_network_encoding(md):
    # create vocabularies and map the ids for multi label columns
    production_indexed, production_vocab_size = vocabularies(md, "production_company_names")
    country_indexed, country_vocab_size = vocabularies(md, "origin_country")
    genre_indexed, genre_vocab_size = vocabularies(md, "genre_names")

    # create vocabulary and map the ids for single label column
    language_vocab = (
        md.select(pl.col("original_language").unique().sort())
        .with_row_index(name="original_language_id")
    )

    language_indexed = (
        md.select(["movie_id", "original_language"])
        .join(language_vocab, on="original_language", how="left")
        .select(["original_language_id"])
    )

    language_vocab_size = len(language_vocab) + 1

    # combine all the data together
    md_neural = pl.concat(
        [
        md.select([ "runtime", "vote_average", "popularity"]),
        production_indexed,
        country_indexed,
        genre_indexed,
        language_indexed
        ],
        how = 'horizontal'
    )

    vocab_sizes = {
        "prod": production_vocab_size,
        "country": country_vocab_size,
        "genre": genre_vocab_size,
        "lang": language_vocab_size
    }

    print("Encoding for neural network models finished")

    return md_neural, vocab_sizes

def preprocess_data():
    # call the function to read and clean the data
    md_clean = read_and_clean_data()

    # call the function to perform linear mini one-hot encoding
    md_linear = linear_mini_hot_encoding(md_clean)

    # call the function to perform neural network encoding
    md_neural, vocab_sizes = neural_network_encoding(md_clean)

    return md_linear, md_neural, vocab_sizes



