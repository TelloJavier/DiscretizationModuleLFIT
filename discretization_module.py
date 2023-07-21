from sklearn.cluster import KMeans
from caimcaim import CAIMD


class DiscretizationModule:
    _METHODS = ['caim', 'kmeans']

    def __init__(self, method):
        if method.lower() in DiscretizationModule._METHODS:
            self.method = method.lower()
        elif method.lower() == 'k-means':
            self.method = 'kmeans'
        else:
            print('Wrong method - These are the supported methods: ' + str(DiscretizationModule._METHODS))

    @staticmethod
    def run_kmeans(data, Q):
        return KMeans(n_clusters=Q, random_state=0).fit(data)

    @staticmethod
    def generar_gray_code(Q):
        if (Q <= 0):
            return [0]

        gray_code = ["0", "1"]

        i = 2
        j = 0

        while(True):

            if (i >= 1 << Q):
                break
            
            for j in range(i-1, -1, -1):
                gray_code.append(gray_code[j])

            for j in range(i):
                gray_code[j] = "0" + gray_code[j]

            for j in range(i, 2*i):
                gray_code[j] = "1" + gray_code[j]

            i = i << 1

        return gray_code


    def fit_transform_and_build_tuples(self, data_df, cont_feats, targets_df, clusters_list_data=None, cont_targets=None, cluster_list_targets=None):
        if self.method == 'kmeans':
            if (clusters_list_data == None) or (cont_targets and (cluster_list_targets == None)):
                print('Discretization with kmeans requires providing the number of clusters to be used')
                return
            return self.fit_transform_and_build_tuples_kmeans(data_df, cont_feats, clusters_list_data, targets_df, cont_targets, cluster_list_targets)
        if self.method == 'caim':
            if (cont_targets):
                print('CAIM needs discrete targets to perform the features discretization')
                return
            return self.fit_transform_and_build_tuples_CAIM(data_df, cont_feats, targets_df)
        else:
            print('Wrong method specified when creating the DiscretizationModule object - These are the supported methods: ' + str(DiscretizationModule._METHODS))
        return

    def fit_transform_and_build_tuples_kmeans(self, data_df, cont_feats, clusters_list_data, targets_df, cont_targets=None, cluster_list_targets=None):
        feat_kmeans = {feat: DiscretizationModule.run_kmeans([[row] for row in data_df.iloc[:,data_df.columns.get_loc(feat)]], clusters_list_data[data_df.columns.get_loc(feat)]) for feat in cont_feats}
        # gray_codes = {feat: generar_gray_code(int(np.ceil(np.log2(clusters_list_data[data_df.columns.get_loc(feat)])))) for feat in cont_feats}

        for feat in cont_feats:
            disc_feat = []
            feat_index = data_df.columns.get_loc(feat)
            for item in data_df.iloc[:,feat_index]:
                cluster_assigned = feat_kmeans[feat].predict([[item]])[0]
                # disc_feat.append(str(gray_codes[feat][cluster_assigned]))
                disc_feat.append(str(cluster_assigned))
            data_df[feat] = disc_feat

        if cont_targets:
            targets_kmeans = {target: DiscretizationModule.run_kmeans([[row] for row in targets_df.iloc[:,targets_df.columns.get_loc(target)]], cluster_list_targets[targets_df.columns.get_loc(target)]) for target in cont_targets}
        
            for target in cont_targets:
                disc_target = []
                target_index = targets_df.columns.get_loc(target)
                for item in targets_df.iloc[:,target_index]:
                    cluster_assigned = targets_kmeans[target].predict([[item]])[0]
                    # disc_target.append(str(gray_codes[target][cluster_assigned]))
                    disc_target.append(str(cluster_assigned))
                targets_df[target] = disc_target

        tuples = list(zip(data_df.values.tolist(), targets_df.values.tolist()))

        return tuples
    
    def fit_transform_and_build_tuples_CAIM(self, data_df, cont_feats, targets_df):
        cont_columns = [data_df.columns.get_loc(feat) for feat in cont_feats]
        data_to_disc = data_df.iloc[:,cont_columns]

        print(data_to_disc.head())

        caim = CAIMD()
        data_disc = caim.fit_transform(data_to_disc, targets_df)

        data_disc = data_disc.astype('int32')
        targets_df = targets_df.astype('int32')

        for feat in cont_feats:
            data_df[feat] = data_disc[feat]

        tuples = list(zip(data_df.values.tolist(), targets_df.values.tolist()))

        return tuples