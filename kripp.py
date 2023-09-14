import pandas as pd
from itertools import chain
from numpy import identity

# Load the CSV file into a DataFrame
annotations_df = pd.read_csv('human_eval.csv')

# Extract ratings from annotators' columns
ratings = annotations_df[['20110183', '20110055', '20110153']].values.tolist()

# Function to calculate agreement table

# Get all agreements
# This function takes annotations {ratings} (as a list) and a set of possible rating label (rating) categories. It calculates an "agreement table" where each row represents the number of annotators who assigned each label to an item. If there are at least two annotators who provided labels for an item, their counts are included in the table. Otherwise, that item is ignored!
def get_annotator_agreement_table(ratings, categories):
    agreement = []
    for annotator_ratings in ratings:
        category_counts = list(map(lambda category: annotator_ratings.count(category), categories))
        if sum(category_counts) > 1:  # Ignore any ratings with less than two annotators
            agreement.append(category_counts)
    return agreement

# Function to calculate weighted annotator count
def get_weighted_annotator_count(weights_k, agreement_i):
    weighted_count = 0
    for i in range(len(agreement_i)):
        weighted_count += weights_k[i] * agreement_i[i]
    return weighted_count

# Get the set of all possible rating categories
rating_categories = set([1, 0, -1])  # Your specific labels
agreement_table = get_annotator_agreement_table(ratings, rating_categories)

n = len(agreement_table)
q = len(rating_categories)

# Get an array with r_i (the total number of annotators who rated the ith comment) for all comments
annotators_per_comment = list(map(lambda r_ik: sum(r_ik), agreement_table))
rhat = sum(annotators_per_comment) / n

# Choose categorical weights
weights = identity(len(rating_categories))

# Calculate observed percent agreement (p_a)
percent_agreement = 0
for i in range(n):  # Find the percent agreement for every comment
    percent_agreement_i = 0
    for k in range(q):  # Find the percent agreement for every category for the ith comment
        rhat_ik = get_weighted_annotator_count(weights[k], agreement_table[i])
        ri = sum(agreement_table[i])  # Number of annotators who rated this comment
        percent_agreement_i_k = (agreement_table[i][k] * (rhat_ik - 1)) / (rhat * (ri - 1))
        percent_agreement_i += percent_agreement_i_k
    percent_agreement += percent_agreement_i

pa_prime = percent_agreement / n  # Find the average comment-level percent agreement
total_annotator_count = n * rhat
pa = (1 - 1 / (total_annotator_count)) * pa_prime + 1 / total_annotator_count

# Calculate classification probabilities
classification_probabilities = [1 / len(rating_categories)] * len(rating_categories)  # Equal probabilities

# Calculate expected percent agreement (p_e)
pe = 0
for k in range(q):  # For every possible pair of rating categories
    for l in range(q):
        # Add the probability of those two categories being chosen at random, weighted by how closely they match
        pe += classification_probabilities[k] * classification_probabilities[l] * weights[k][l]

# Calculate Krippendorff's Alpha
alpha = (pa - pe) / (1 - pe)
print(f"Krippendorff's alpha: {alpha}")
