1. Утановить dvc ( ```pip install dvc``` )
2. Почитать введение в dvc ( https://dvc.org/doc/start )
3. Пройти tutorial https://dvc.org/doc/use-cases/versioning-data-and-model-files/tutorial
    1. ```git clone https://github.com/iterative/example-versioning.git```
    2. ```cd example-versioning```
    3. ```pip install -r requirements.txt```
    4. ```dvc get https://github.com/iterative/dataset-registry tutorials/versioning/data.zip```
    5. ```unzip -q data.zip```
    6. ```rm -f data.zip```
    7. ``` ls -l -R| less```
    8. ```python train.py```
    9. ```dvc add model.h5```
    10. ```git add data.dvc model.h5.dvc metrics.csv .gitignore```
    11. ```git commit -m "First model, trained with 1000 images"``` (Что может пойти не так?)
    12. ```git tag -a "v1.0" -m "model v1.0, 1000 images"```
    13. ```git status``` (Что за файлы вне git?)
    14. ```dvc get https://github.com/iterative/dataset-registry tutorials/versioning/new-labels.zip```
    15. ```unzip -q new-labels.zip```
    16. ```rm -f new-labels.zip```
    17. ```python train.py```
    18. ```dvc add model.h5```
    19. ```git add data.dvc model.h5.dvc metrics.csv```
    20. ```git commit -m "Second model, trained with 2000 images"```
    21. ```git tag -a "v2.0" -m "model v2.0, 2000 images"```
    22. Переключение между кодом и даннымми версий 1.0 и 2.0
    23. ```dvc remove model.h5.dvc```
    24. Автоматическое отслеживание промежуточных файлов и метрик
```
dvc run -n train -d train.py -d data \
          -o model.h5 -o bottleneck_features_train.npy \
          -o bottleneck_features_validation.npy -M metrics.csv \
          python train.py
```
4. Изучить понятие dvc experiment ( https://dvc.org/doc/start/experiments )