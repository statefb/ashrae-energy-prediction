参考：https://upura.hatenablog.com/entry/2018/12/28/225234
https://github.com/ghmagazine/kagglebook/tree/master/ch04-model-interface/code

* configs, runs, submission, logsは合致させる。submissionsとlogsは自動ネーミング
    - ex)
    - configs/baseline.json
    - configs/baseline.py
    - configs/sub_(19-10-31-32)_(1.45).py
    - configs/log_(19-10-31-32).log
* config管理：
    - どの特徴量？
    - どのモデル？
    - パラメータ？
    - （どのバリデーション）？
* バリデーション：Runner.load_index_fold内に記載

data
	raw
		train.csv
		test.csv
	interim
	processed
        train.feather
		test.feather
	external

configs
	default.json

features
	familiy_size_train.feather

src
	data
		preprocess
			191025_drop-na.py
			...
	features
		
	models
		model.py
		model_knn.py
		...
	runner.py
	util.py - contextmanager timer, convert_to_feather
	runs
		191025_knn-with-zero.py
		...
submission
	sub_(year-month-day-hour-min)_(score).csv
logs
	log_(year-month-day-hour-min).log

memo
	discussions.md
models
	model_knn.feather
notebooks
	eda.ipynb
(references)

----------------------------

default.json:

{
    "features": [
        "age",
        "embarked",
        "family_size",
        "fare",
        "pclass",
        "sex"
    ],
    "lgbm_params": {
        "learning_rate": 0.1,
        "num_leaves": 8,
        "boosting_type": "gbdt",
        "colsample_bytree": 0.65,
        "reg_alpha": 1,
        "reg_lambda": 1,
        "objective": "multiclass",
        "num_class": 2
    },
    "loss": "multi_logloss",
    "target_name": "Survived",
    "ID_name": "PassengerId"
}