from onecodex import Api
ocx = Api()

project = ocx.Projects.get("d53ad03b010542e3")  # get DIABIMMUNE project by ID
samples = ocx.Samples.where(project=project.id, public=True, limit=500)

samples.metadata[[
    "gender",
    "host_age",
    "geo_loc_name",
    "totalige",
    "eggs",
    "vegetables",
    "milk",
    "wheat",
    "rice",
]].head(20)


chao1 = samples.plot_metadata(vaxis="chao1", haxis="geo_loc_name", return_chart=True)
simpson = samples.plot_metadata(vaxis="simpson", haxis="geo_loc_name", return_chart=True)
shannon = samples.plot_metadata(vaxis="shannon", haxis="geo_loc_name", return_chart=True)


samples.plot_metadata(haxis="host_age", vaxis="Bacteroides", plot_type="scatter")


# generate a dataframe containing relative abundances (default metric for WGS datasets)
df_rel = samples.to_df()

# fetch all samples for subject P014839
subject_metadata = samples.metadata.loc[samples.metadata["host_subject_id"] == "P014839"]
subject_df = df_rel.loc[subject_metadata.index]

# order by subject age
subject_df = subject_df.loc[subject_metadata["host_age"].sort_values().index]

# you can access our library using the ocx accessor on pandas dataframes!
subject_df.ocx.plot_bargraph(
    rank="genus",
    label=lambda metadata: str(metadata["host_age"]),
    title="Subject P014839 Over Time",
    xlabel="Host Age at Sampling Time (days)",
    ylabel="Normalized Read Count",
    legend="Genus",
    
    
f_rel[:30].ocx.plot_heatmap(legend="Normalized Read Count", tooltip="geo_loc_name")
    
    
# generate a dataframe containing absolute read counts
df_abs = samples.to_df(normalize=False)
df_abs[:30].ocx.plot_distance(metric="weighted_unifrac")


samples.plot_pca(color="host_age", size="Bifidobacterium")


samples.plot_mds(
    metric="weighted_unifrac", method="pcoa", color="geo_loc_name"
)
