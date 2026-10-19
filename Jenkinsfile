// Potok ML: dane (DVC) -> trening -> ewaluacja -> próg jakości -> rejestracja w MLflow -> wdrożenie API
pipeline {
    agent any

    options {
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '20'))
        disableConcurrentBuilds()
    }

    parameters {
        string(name: 'EPOCHS', defaultValue: '5', description: 'Liczba epok')
        string(name: 'LEARNING_RATE', defaultValue: '2e-5', description: 'Współczynnik uczenia')
        choice(name: 'LR_SCHEDULER', choices: ['linear', 'cosine'], description: 'Harmonogram współczynnika uczenia')
        string(name: 'WEIGHT_DECAY', defaultValue: '0.01', description: 'Weight decay (np. 0.01, 0.08)')
        string(name: 'EARLY_STOPPING_PATIENCE', defaultValue: '2', description: 'Early stopping: liczba epok bez poprawy eval_loss (0 = wyłączone)')
        string(name: 'BATCH_SIZE', defaultValue: '16', description: 'Rozmiar batcha (GTX 1660 Ti: maks. 16–32)')
        string(name: 'SAMPLE_SIZE', defaultValue: '0', description: 'Liczba próbek (0 = cały zbiór)')
        string(name: 'MIN_F1', defaultValue: '0.80', description: 'Minimalne F1 (ważone) na zbiorze testowym')
        booleanParam(name: 'USE_GPU', defaultValue: true, description: 'Trening na GPU (--gpus all)')
        booleanParam(name: 'REGISTER', defaultValue: true, description: 'Rejestracja modelu w MLflow Model Registry')
        booleanParam(name: 'DEPLOY', defaultValue: true, description: 'Wdrożenie API z nowym modelem')
    }

    environment {
        DAGSHUB_REPO = 'irintinuviel/crypto_sentiment'
        MLFLOW_TRACKING_URI = 'https://dagshub.com/irintinuviel/crypto_sentiment.mlflow'
        MLFLOW_EXPERIMENT_NAME = 'jenkins'
    }

    stages {
        stage('Potok ML') {
            agent {
                dockerfile {
                    filename 'Dockerfile.train'
                    additionalBuildArgs "--build-arg TORCH_INDEX=https://download.pytorch.org/whl/${params.USE_GPU ? 'cu121' : 'cpu'}"
                    args "${params.USE_GPU ? '--gpus all' : ''} --shm-size=2g"
                    reuseNode true
                }
            }
            environment {
                HF_HOME = "${WORKSPACE}/.cache/huggingface"
            }
            stages {
                stage('Dane (DVC)') {
                    steps {
                        withCredentials([usernamePassword(credentialsId: 'dagshub',
                                usernameVariable: 'DAGSHUB_USER', passwordVariable: 'DAGSHUB_TOKEN')]) {
                            sh '''
                                dvc remote modify origin --local auth basic
                                dvc remote modify origin --local user "$DAGSHUB_USER"
                                dvc remote modify origin --local password "$DAGSHUB_TOKEN"
                                dvc pull data/final_crypto_data.csv.dvc
                            '''
                        }
                    }
                }
                stage('Parametry') {
                    steps {
                        sh """
                            python -m src.params \
                                train.epochs=${params.EPOCHS} \
                                train.learning_rate=${params.LEARNING_RATE} \
                                train.lr_scheduler=${params.LR_SCHEDULER} \
                                train.weight_decay=${params.WEIGHT_DECAY} \
                                train.early_stopping_patience=${params.EARLY_STOPPING_PATIENCE} \
                                train.batch_size=${params.BATCH_SIZE} \
                                data.sample_size=${params.SAMPLE_SIZE}
                        """
                    }
                }
                stage('Trening i ewaluacja (dvc repro)') {
                    steps {
                        withCredentials([usernamePassword(credentialsId: 'dagshub',
                                usernameVariable: 'MLFLOW_TRACKING_USERNAME', passwordVariable: 'MLFLOW_TRACKING_PASSWORD')]) {
                            sh 'python -c "import torch; print(\'CUDA:\', torch.cuda.is_available())"'
                            sh 'echo "MLflow: $MLFLOW_TRACKING_URI, eksperyment: $MLFLOW_EXPERIMENT_NAME"'
                            sh 'dvc repro'
                            sh 'dvc metrics show'
                        }
                    }
                }
                stage('Próg jakości') {
                    steps {
                        sh "python -m src.check_metrics --min ${params.MIN_F1}"
                    }
                }
                stage('Rejestracja modelu (MLflow)') {
                    when { expression { params.REGISTER } }
                    steps {
                        withCredentials([usernamePassword(credentialsId: 'dagshub',
                                usernameVariable: 'MLFLOW_TRACKING_USERNAME', passwordVariable: 'MLFLOW_TRACKING_PASSWORD')]) {
                            sh 'python -m src.register'
                        }
                    }
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'reports/**, params.yaml, dvc.lock', allowEmptyArchive: true
                }
            }
        }

        stage('Wdrożenie API') {
            when { expression { params.DEPLOY } }
            steps {
                sh '''
                    docker build -t crypto-sentiment-api:${BUILD_NUMBER} -t crypto-sentiment-api:latest .
                    docker network create crypto-net || true
                    docker rm -f crypto-api || true
                    docker create --name crypto-api --network crypto-net -p 8000:8000 crypto-sentiment-api:${BUILD_NUMBER}
                    docker cp models/model crypto-api:/app/model
                    docker start crypto-api
                '''
                // test dymny wdrożonego API
                sh '''
                    for i in $(seq 1 30); do
                        docker exec crypto-api python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" && break
                        sleep 5
                    done
                    docker exec crypto-api python -c "
import json, urllib.request
req = urllib.request.Request('http://localhost:8000/predict', data=json.dumps({'texts': ['bitcoin to the moon']}).encode(), headers={'Content-Type': 'application/json'})
print(urllib.request.urlopen(req).read().decode())
"
                '''
            }
        }
    }

    post {
        success {
            echo 'Model wytrenowany, zarejestrowany i wdrożony: http://localhost:8000/docs'
        }
    }
}
