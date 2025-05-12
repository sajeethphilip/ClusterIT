pip install pandas hdbscan umap-learn plotly scikit-learn ipywidgets kaleido

pip install pandas hdbscan scikit-learn bayesian-optimization

pip install pandas hdbscan scikit-learn plotly kaleido
```
(my-python) nsp@ninan-latitudee5570:~/Downloads/ML/DBNN/IDBNN/ClusterIT$ python autoclusterIT.py 
Enter CSV file path: powspec3/powspec3.csv
Enter label column (or Enter to skip): label
|   iter    |  target   | min_cl... | min_sa... |
-------------------------------------------------
| 1         | 0.07133   | 19.98     | 19.06     |
| 2         | 0.05726   | 37.14     | 12.37     |
| 3         | 0.03748   | 9.489     | 3.964     |
| 4         | 0.0223    | 4.788     | 17.46     |
| 5         | 0.01924   | 30.85     | 14.45     |
| 6         | 0.03477   | 19.95     | 18.94     |
| 7         | 0.01814   | 41.77     | 6.705     |
| 8         | 0.02044   | 6.865     | 10.91     |
| 9         | 0.05867   | 2.271     | 9.812     |
| 10        | 0.01933   | 34.04     | 13.2      |
| 11        | -0.002636 | 3.559     | 5.09      |
| 12        | 0.02159   | 23.21     | 16.66     |
| 13        | 0.07133   | 20.03     | 19.16     |
| 14        | 0.03748   | 9.541     | 3.96      |
| 15        | 0.07133   | 19.84     | 19.12     |
| 16        | 0.02257   | 30.99     | 3.882     |
| 17        | 0.07133   | 19.58     | 19.19     |
| 18        | 0.07133   | 19.51     | 19.31     |
| 19        | 0.07133   | 19.75     | 19.37     |
| 20        | 0.05867   | 2.216     | 9.86      |
=================================================


Optimal parameters found:
min_cluster_size: 20
min_samples: 19
Saved results to powspec3/powspec3_auto_clustered.csv

```
