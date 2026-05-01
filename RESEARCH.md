# Research Notes For The Price Predictor

This project is now restricted to the following input columns:

- `make`
- `model`
- `year`
- `variant`
- `mileage`
- `transmission`
- `fuel_type`
- `inspection_score`
- `registered_in`
- `color`
- `assembly`

Target:

- `price`

## Recommendation

Train a gradient-boosted tree regressor that natively handles categorical features.

Recommended first model:

- `CatBoostRegressor` for point prediction
- `CatBoostRegressor(loss_function="Quantile:alpha=0.10")` for lower bound
- `CatBoostRegressor(loss_function="Quantile:alpha=0.90")` for upper bound

Reason:

- your schema is mostly categorical and high-cardinality
- `make`, `model`, and especially `variant` are difficult for plain linear models
- `inspection_score` likely interacts non-linearly with `mileage` and `year`
- tree boosting handles these interactions with much less manual encoding

## What the research suggests

### 1. Native categorical handling strongly matters

CatBoost documentation says not to one-hot encode categorical features during preprocessing, and explains that categorical features are internally transformed using target/statistical encodings and combinations. That makes it a good fit for `make`, `model`, `variant`, `registered_in`, `color`, and `assembly`.

Sources:

- [CatBoost categorical features](https://catboost.ai/docs/en/features/categorical-features)
- [CatBoost transforming categorical features](https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic)

### 2. Quantile models are appropriate for price ranges

CatBoost officially supports `Quantile` and `MultiQuantile` regression objectives, so a price range is not a hack. It is a supported regression setup.

Source:

- [CatBoost regression objectives and metrics](https://catboost.ai/docs/en/concepts/loss-functions-regression)

### 3. Tree boosting is the right baseline for this feature mix

LightGBM and XGBoost both support categorical handling, but LightGBM warns that very high-cardinality categorical features can need special treatment, and XGBoost categorical support is still more hands-on from a pipeline perspective. For this schema, CatBoost is the simplest strong baseline.

Sources:

- [LightGBM advanced topics: categorical feature support](https://lightgbm.readthedocs.io/en/v4.5.0/Advanced-Topics.html)
- [XGBoost categorical data tutorial](https://xgboost.readthedocs.io/en/latest/tutorials/categorical.html)

### 4. Public used-car projects keep converging on tree ensembles

Public GitHub projects regularly compare Random Forest, XGBoost, LightGBM, and CatBoost for used-car pricing. One such repo reports the best result from an ensemble of boosted/tree models, with CatBoost, LightGBM, and XGBoost all among the strongest individual candidates.

Source:

- [YashsTiwari/Used-Car-Price-Prediction](https://github.com/YashsTiwari/Used-Car-Price-Prediction)

This is not primary evidence, but it is directionally consistent with the official library capabilities.

### 5. Year and mileage repeatedly show up as top features

Older public used-car projects and papers repeatedly report that `year` and `mileage` are among the strongest signals. One GitHub analysis explicitly highlights Year and Mileage as important to price, and a Random Forest paper also argues that feature selection around the most correlated variables drives usable accuracy.

Sources:

- [francescopisu/UsedCarPricePrediction](https://github.com/francescopisu/UsedCarPricePrediction)
- [How much is my car worth?](https://arxiv.org/abs/1711.06970)

## How to think about your allowed features

This ranking is an inference from the sources above plus domain logic:

### Highest expected signal

- `year`
- `mileage`
- `make`
- `model`
- `variant`
- `inspection_score`

### Medium expected signal

- `assembly`
- `registered_in`
- `transmission`
- `fuel_type`

### Lower but still usable signal

- `color`

`color` should stay in the dataset, but you should not expect it to move accuracy as much as `year`, `mileage`, `variant`, or `inspection_score`.

## Best training strategy for this exact schema

Use all allowed features, but let the model decide importance.

Do not manually drop `color`, `assembly`, or `registered_in` before you test them. Instead:

1. Train with all allowed columns.
2. Save feature importance.
3. Compare cross-validated MAE and MAPE with and without weaker features.

That is why the training script now exports `feature_importance.csv`.

## What will make the model truly accurate

The model choice matters, but data discipline matters more:

- use sold prices, not asking prices
- use one market at a time if possible
- standardize `variant` names carefully
- keep `inspection_score` on a stable 0-10 rubric
- normalize `registered_in`
- normalize `assembly`
- normalize `color`
- remove obvious outliers
- keep the training data recent

## Practical conclusion

If you only allow the 11 input fields above, the best first production candidate is CatBoost with:

- all 11 features included
- log-price target
- quantile side-models for range prediction
- saved feature importance after each training run

That is the configuration now implemented in this project.

## Inspection architecture

For better real-world precision, inspection should be treated as a flexible subsystem:

- if user already has `inspection_score`, use it
- if user does not have it, compute it from weighted sections
- if section-level inputs are available, feed both the overall score and the section signals into the model
- keep body-frame accident data as a separate critical feature instead of collapsing it entirely into the overall score

This is stricter and more production-grade than forcing every user to provide a full inspection report up front.
