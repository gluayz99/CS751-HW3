from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)


def data_preparation():
    df = pd.read_csv('data/snsdata.csv', encoding='utf-8')
    df['age'].interpolate(axis=0, method='nearest', limit_direction='both', limit=10, inplace=True)
    df['gender'].bfill(axis=0, inplace=True)
    df['gender'] = df['gender'].replace(['M', 'F'], [0, 1])

    return df


def define_person_type(row):
    max_ss = -1
    curr_type = -1
    if row['extra'] > max_ss:
        max_ss = row['extra']
        curr_type = 1

    if row['fashion'] > max_ss:
        max_ss = row['fashion']
        curr_type = 2

    if row['religion'] > max_ss:
        max_ss = row['religion']
        curr_type = 3

    if row['romance'] > max_ss:
        max_ss = row['romance']
        curr_type = 4

    if row['anti'] > max_ss:
        max_ss = row['anti']
        curr_type = 5

    return curr_type


def data_preprocessing(df):
    col_extra = ['basketball', 'football', 'soccer', 'softball', 'volleyball', 'swimming', 'cheerleading', 'baseball',
                 'tennis', 'sports', 'dance', 'band', 'marching', 'music', 'rock']
    col_fashion = ['hair', 'dress', 'blonde', 'mall', 'shopping', 'clothes']
    col_religion = ['god', 'church', 'jesus', 'bible']
    col_romance = ['cute', 'sex', 'sexy', 'hot', 'kissed']
    col_anti = ['hollister', 'abercrombie', 'die', 'death', 'drunk', 'drugs']

    col_info = ['gradyear', 'gender', 'age', 'friends']
    df = df.drop(col_info, axis=1)
    # df['age'] = df['age'].astype(float).round().astype(int)

    df['extra'] = df[col_extra].sum(axis=1)
    df = df.drop(col_extra, axis=1)
    #
    df['fashion'] = df[col_fashion].sum(axis=1)
    df = df.drop(col_fashion, axis=1)
    #
    df['religion'] = df[col_religion].sum(axis=1)
    df = df.drop(col_religion, axis=1)
    #
    df['romance'] = df[col_romance].sum(axis=1)
    df = df.drop(col_romance, axis=1)
    #
    df['anti'] = df[col_anti].sum(axis=1)
    df = df.drop(col_anti, axis=1)

    # df['person_type'] = df.apply(define_person_type, axis=1)
    print('Describe')
    print(df.describe())
    print('-' * 20, '\n\n')

    print('Correlation')
    print(df.corr())
    print('-' * 20, '\n\n')

    print('Detail')
    print(df)
    print(df.dtypes)
    print('-' * 20, '\n\n')

    return df


def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse


def data_processing(df):
    # ['extra', 'fashion', 'religion', 'romance', 'anti']
    x = df.iloc[:].values

    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(x))
    scaled_features.columns = df.columns

    wcss = []
    wcss_range = range(1, 30)
    for i in wcss_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init='auto', random_state=69)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

    plt.plot(wcss_range, wcss)
    plt.title('The elbow method')
    plt.xticks(wcss_range)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')  # within cluster sum of squares
    plt.show()

    # sil = []
    # sil_range = range(2, 11)
    # for i in sil_range:
    #     kmeans = KMeans(n_clusters=i, max_iter=300, n_init='auto', random_state=69)
    #     kmeans.fit(scaled_features)
    #     labels = kmeans.labels_
    #     sil.append(silhouette_score(x, labels, metric='euclidean'))
    #
    # plt.plot(sil_range, sil)
    # plt.title('The Silhouette Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette score')  # within cluster sum of squares
    # plt.show()


def evaluate_kmeans(n_clusters, df):
    x = df.iloc[:].values
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(x))
    scaled_features.columns = df.columns

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=69)
    y_kmeans = kmeans.fit_predict(scaled_features)
    df['cluster'] = kmeans.labels_

    for i in range(n_clusters):
        _y_kmeans = list(y_kmeans)
        print(f'class {i + 1}: {_y_kmeans.count(i)}')

    color = iter(cm.rainbow(np.linspace(0, 1, n_clusters + 1)))
    for i in range(n_clusters):
        plt.scatter(x[y_kmeans == i, 0], x[y_kmeans == i, 1], s=50, c=next(color), label=str(i))
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=20, c=next(color), label='Centroids')
    plt.legend()
    plt.show()

    return df


def data_analyst(n_clusters, df, df_original):
    col_info = ['gradyear', 'gender', 'age', 'friends']
    df_original = df_original[col_info]

    df = pd.concat([df, df_original], axis=1)

    print(df.groupby('cluster').agg({'cluster': 'count'}))
    print('-' * 25, '\n\n')
    for i in range(n_clusters):
        print(f'Cluster Group {i}')
        rslt_df = df[df['cluster'] == i].copy()
        print(rslt_df.head())
        print(rslt_df.describe())
        print('-' * 10, '\n')

        rslt_df['person_type'] = rslt_df.apply(define_person_type, axis=1)
        print(print(rslt_df.groupby('person_type').agg({'person_type': 'count'})))
        print('=' * 20, '\n\n')


def main():
    df_original = data_preparation()

    df = data_preprocessing(df_original)

    data_processing(df)

    n_clusters = int(input('Input the optimal clusters: '))
    df = evaluate_kmeans(n_clusters, df)

    data_analyst(n_clusters, df, df_original)


if __name__ == "__main__":
    main()
