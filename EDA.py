import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F

class EDA():
    def __init__(self, target_ctr):
        self.target_ctr = target_ctr

    def id_analysis(self, raw_df, id_col):
        """
        plotting out distribution of id column and return a list of high CTR ids

        Parameters
        ----------
        id_col str:
            the id column of primary feature

        Returns
        -------
        list
            a list of high CTR ids
        """

        df = raw_df.select(id_col, "label_int").groupBy(id_col).agg({"*": "count", "label_int": "sum"}).toPandas()
        df = df.set_index(id_col)

        # CTR of each ID
        df[f"{id_col}_ctr"] = df.apply(lambda row: row[1] / row[0], axis=1)
        print(f"CTR distribution by {id_col}")
        print(df[f"{id_col}_ctr"].describe())

        # dist plot
        sns.distplot(df[f"{id_col}_ctr"])
        plt.show()
        # cumulative plot
        sns.kdeplot(df[f"{id_col}_ctr"], cumulative=True)
        plt.show()

        # Bin CTR to decide high or low CTR in this category
        df["ctr_qbin"] = pd.qcut(df[f"{id_col}_ctr"], 4)
        qbin_df = df.groupby("ctr_qbin").sum()
        (qbin_df["sum(label_int)"] / qbin_df["count(1)"]).plot.bar()
        plt.show()

        # see how is high ctr group performing
        sns.boxplot(x="ctr_qbin", y=f"{id_col}_ctr", data=df)
        plt.show()
        # define ctr over 3rd quartile is high ctr
        high_ctr_ids = df[df[f"{id_col}_ctr"] >= df[f"{id_col}_ctr"].quantile(0.75)].index
        low_ctr_ids = df[df[f"{id_col}_ctr"] < df[f"{id_col}_ctr"].quantile(0.75)].index
        return (list(high_ctr_ids), list(low_ctr_ids))

    def cat_analysis(self,raw_df,col,prime_id):
        """
        plotting out the CTR distribution of each category

        Parameters
        ----------
        raw_df spark dataframe:
            the spark data frame to query data
        col str:
            the column want to analyze
        prime_id:
            id of major feature

        Returns
        -------
        None
        """

        dist = raw_df[prime_id,col].groupBy(col).agg(F.countDistinct(prime_id)).toPandas()
        dist.set_index(col).sort_index().plot.bar()
        plt.show()
        count = raw_df.select(col).groupBy(col).count().toPandas().set_index(col)
        click = raw_df.select(col,"label_int").groupBy(col).sum("label_int").toPandas().set_index(col)
        print("CTR")
        (click.iloc[:,0]/count.iloc[:,0]).sort_index().plot.bar()
        plt.hlines(self.target_ctr, -1, 1000,linestyles='dashed',colors="r")
        plt.show()

    def high_low_ctr_group_dist(self, raw_df, col, prime_id, high_ctr_tags):
        """
        plotting out the distribution of high low ctr group in specific column

        Parameters
        ----------
        raw_df spark dataframe:
            the spark data frame to query data
        col str:
            the column want to analyze
        prime_id str:
            id of major feature
        high_ctr_tags list:
             the ids of high ctr group

        Returns
        -------
        None
        """
        df = raw_df.select(prime_id, col, "label_int").groupBy(prime_id, col).agg({prime_id: "count", "label_int": "sum"}).toPandas()
        df["ctr_group"] = df[prime_id].apply(lambda x: "high_ctr" if x in high_ctr_tags else "low_ctr")
        agg_df = df.groupby([col,"ctr_group"]).sum()
        agg_df = agg_df.reset_index()
        high = agg_df[agg_df["ctr_group"] == "high_ctr"]
        high = high.set_index(col)["count(1)"].sum()
        low = agg_df[agg_df["ctr_group"] == "low_ctr"]
        low = low.set_index(col)["count(1)"].sum()
        agg_df["count_pct"] = np.where(agg_df["ctr_group"] == "high_ctr",agg_df["count(1)"]/high,agg_df["count(1)"]/low)
        print("High Low CTR Group Distribution")
        sns.barplot(x=col,y="count_pct",hue="ctr_group",data=agg_df)
        plt.show()