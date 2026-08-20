# Fargate Deployment Guide

Deploy model training & scoring to AWS ECS Fargate — serverless, pay-per-second, unlimited runtime.

---

## Architecture

```
EventBridge (cron scheduler)
        ↓
Lambda function (trigger)
        ↓
ECS Fargate Task (runs container)
        ↓
Docker container (train_model.py or predict.py)
        ↓
S3 (save model/predictions)
        ↓
CloudWatch Logs (monitor)
```

**Workflows:**
- **Weekly:** Score open deals (2 min)
- **Monthly:** Calibrate model (5 min)
- **Yearly:** Full retrain (20 min)

---

## Prerequisites

```bash
# AWS CLI configured
aws configure

# Docker installed
docker --version

# AWS Account permissions
# - ECR (push images)
# - ECS (run tasks)
# - Lambda (trigger tasks)
# - EventBridge (schedule)
# - IAM (roles)
# - S3 (store data/models)
# - CloudWatch (logs)
```

---

## Step 1: Create Docker Image

### 1.1 Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY scripts/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts
COPY scripts/train_model.py .
COPY scripts/predict.py .
COPY scripts/build_features.py .

# AWS CLI for S3 operations
RUN pip install awscli

# Set entry point (can be overridden)
ENTRYPOINT ["python"]
CMD ["predict.py"]
```

### 1.2 Build Image

```bash
# Build locally
docker build -t forecast-model:latest .

# Test locally
docker run \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION=us-west-2 \
  forecast-model:latest predict.py
```

---

## Step 2: Push to ECR

### 2.1 Create ECR Repository

```bash
REGION=us-west-2
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

aws ecr create-repository \
  --repository-name forecast-model \
  --region $REGION
```

### 2.2 Push Image

```bash
# Login to ECR
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin \
  $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

# Tag image
docker tag forecast-model:latest \
  $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/forecast-model:latest

# Push
docker push \
  $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/forecast-model:latest
```

---

## Step 3: Setup IAM Roles

### 3.1 ECS Task Execution Role

```bash
# Create trust policy
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document file://trust-policy.json

# Attach policy
aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

### 3.2 ECS Task Role (S3 + Logs)

```bash
# Create trust policy for task
cat > task-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name ecsTaskRole \
  --assume-role-policy-document file://task-trust-policy.json

# Create policy for S3 + CloudWatch
cat > task-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::forecast-bucket",
        "arn:aws:s3:::forecast-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
EOF

# Attach policy
aws iam put-role-policy \
  --role-name ecsTaskRole \
  --policy-name S3CloudWatchPolicy \
  --policy-document file://task-policy.json
```

### 3.3 Lambda Execution Role

```bash
# Create trust policy
cat > lambda-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name lambdaECSRole \
  --assume-role-policy-document file://lambda-trust-policy.json

# Policy to run ECS tasks
cat > lambda-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "ecs:RunTask",
      "Resource": "arn:aws:ecs:*:$ACCOUNT:task-definition/forecast-*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": [
        "arn:aws:iam::$ACCOUNT:role/ecsTaskExecutionRole",
        "arn:aws:iam::$ACCOUNT:role/ecsTaskRole"
      ]
    }
  ]
}
EOF

aws iam put-role-policy \
  --role-name lambdaECSRole \
  --policy-name RunECSTasksPolicy \
  --policy-document file://lambda-policy.json
```

---

## Step 4: Create ECS Cluster & Task Definition

### 4.1 Create Cluster

```bash
aws ecs create-cluster \
  --cluster-name forecast-cluster \
  --region $REGION
```

### 4.2 Task Definition for Scoring

```bash
# Save as task-def-score.json
cat > task-def-score.json <<EOF
{
  "family": "forecast-score",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "taskRoleArn": "arn:aws:iam::$ACCOUNT:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::$ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "forecast",
      "image": "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/forecast-model:latest",
      "command": ["predict.py"],
      "environment": [
        {
          "name": "S3_BUCKET",
          "value": "forecast-bucket"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/forecast-score",
          "awslogs-region": "$REGION",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOF

# Register task definition
aws ecs register-task-definition \
  --cli-input-json file://task-def-score.json \
  --region $REGION
```

### 4.3 Task Definition for Calibration

```bash
cat > task-def-calibrate.json <<EOF
{
  "family": "forecast-calibrate",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "taskRoleArn": "arn:aws:iam::$ACCOUNT:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::$ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "forecast",
      "image": "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/forecast-model:latest",
      "command": ["train_model.py", "--calibrate-only"],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/forecast-calibrate",
          "awslogs-region": "$REGION",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOF

aws ecs register-task-definition \
  --cli-input-json file://task-def-calibrate.json \
  --region $REGION
```

### 4.4 Task Definition for Retrain

```bash
cat > task-def-retrain.json <<EOF
{
  "family": "forecast-retrain",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "taskRoleArn": "arn:aws:iam::$ACCOUNT:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::$ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "forecast",
      "image": "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/forecast-model:latest",
      "command": ["train_model.py"],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/forecast-retrain",
          "awslogs-region": "$REGION",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOF

aws ecs register-task-definition \
  --cli-input-json file://task-def-retrain.json \
  --region $REGION
```

---

## Step 5: Create Lambda Functions

### 5.1 Lambda for Weekly Scoring

```bash
# Save as lambda_score.py
cat > lambda_score.py <<'EOF'
import boto3
import json
from datetime import datetime

ecs = boto3.client('ecs')

def lambda_handler(event, context):
    """Trigger weekly forecast scoring"""
    
    response = ecs.run_task(
        cluster='forecast-cluster',
        taskDefinition='forecast-score:1',
        launchType='FARGATE',
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': ['subnet-12345678'],  # Replace with your subnet
                'securityGroups': ['sg-12345678'],  # Replace with your security group
                'assignPublicIp': 'ENABLED'
            }
        }
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps(f"Started task: {response['tasks'][0]['taskArn']}")
    }
EOF

# Package and upload
zip lambda_score.zip lambda_score.py

aws lambda create-function \
  --function-name forecast-score \
  --runtime python3.9 \
  --role arn:aws:iam::$ACCOUNT:role/lambdaECSRole \
  --handler lambda_score.lambda_handler \
  --zip-file fileb://lambda_score.zip \
  --region $REGION
```

### 5.2 Lambda for Monthly Calibration

```bash
cat > lambda_calibrate.py <<'EOF'
import boto3

ecs = boto3.client('ecs')

def lambda_handler(event, context):
    """Trigger monthly calibration"""
    
    response = ecs.run_task(
        cluster='forecast-cluster',
        taskDefinition='forecast-calibrate:1',
        launchType='FARGATE',
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': ['subnet-12345678'],
                'securityGroups': ['sg-12345678'],
                'assignPublicIp': 'ENABLED'
            }
        }
    )
    
    return {
        'statusCode': 200,
        'body': f"Calibration task started: {response['tasks'][0]['taskArn']}"
    }
EOF

zip lambda_calibrate.zip lambda_calibrate.py

aws lambda create-function \
  --function-name forecast-calibrate \
  --runtime python3.9 \
  --role arn:aws:iam::$ACCOUNT:role/lambdaECSRole \
  --handler lambda_calibrate.lambda_handler \
  --zip-file fileb://lambda_calibrate.zip \
  --region $REGION
```

---

## Step 6: Setup EventBridge Cron Rules

### 6.1 Weekly Scoring (Monday 9am UTC)

```bash
# Create rule
aws events put-rule \
  --name weekly-forecast-score \
  --schedule-expression "cron(0 9 ? * MON *)" \
  --state ENABLED \
  --region $REGION

# Add Lambda target
aws events put-targets \
  --rule weekly-forecast-score \
  --targets "Id"="1","Arn"="arn:aws:lambda:$REGION:$ACCOUNT:function:forecast-score","RoleArn"="arn:aws:iam::$ACCOUNT:role/lambdaECSRole" \
  --region $REGION
```

### 6.2 Monthly Calibration (1st of month 9am UTC)

```bash
aws events put-rule \
  --name monthly-forecast-calibrate \
  --schedule-expression "cron(0 9 1 * ? *)" \
  --state ENABLED \
  --region $REGION

aws events put-targets \
  --rule monthly-forecast-calibrate \
  --targets "Id"="1","Arn"="arn:aws:lambda:$REGION:$ACCOUNT:function:forecast-calibrate","RoleArn"="arn:aws:iam::$ACCOUNT:role/lambdaECSRole" \
  --region $REGION
```

### 6.3 Yearly Retrain (Jan 1 9am UTC)

```bash
# Create Lambda for retrain first
cat > lambda_retrain.py <<'EOF'
import boto3

ecs = boto3.client('ecs')

def lambda_handler(event, context):
    response = ecs.run_task(
        cluster='forecast-cluster',
        taskDefinition='forecast-retrain:1',
        launchType='FARGATE',
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': ['subnet-12345678'],
                'securityGroups': ['sg-12345678'],
                'assignPublicIp': 'ENABLED'
            }
        }
    )
    return {'statusCode': 200}
EOF

# Create function
zip lambda_retrain.zip lambda_retrain.py
aws lambda create-function \
  --function-name forecast-retrain \
  --runtime python3.9 \
  --role arn:aws:iam::$ACCOUNT:role/lambdaECSRole \
  --handler lambda_retrain.lambda_handler \
  --zip-file fileb://lambda_retrain.zip \
  --region $REGION

# Create rule
aws events put-rule \
  --name yearly-forecast-retrain \
  --schedule-expression "cron(0 9 1 1 ? *)" \
  --state ENABLED \
  --region $REGION

aws events put-targets \
  --rule yearly-forecast-retrain \
  --targets "Id"="1","Arn"="arn:aws:lambda:$REGION:$ACCOUNT:function:forecast-retrain","RoleArn"="arn:aws:iam::$ACCOUNT:role/lambdaECSRole" \
  --region $REGION
```

---

## Step 7: Monitor & Troubleshoot

### View Logs

```bash
# Weekly scoring logs
aws logs tail /ecs/forecast-score --follow

# Monthly calibration logs
aws logs tail /ecs/forecast-calibrate --follow

# Yearly retrain logs
aws logs tail /ecs/forecast-retrain --follow
```

### Check Task Status

```bash
# List running tasks
aws ecs list-tasks --cluster forecast-cluster

# Describe task
aws ecs describe-tasks \
  --cluster forecast-cluster \
  --tasks arn:aws:ecs:$REGION:$ACCOUNT:task/forecast-cluster/abc123...
```

### Trigger Manually

```bash
# Test weekly scoring
aws lambda invoke \
  --function-name forecast-score \
  --region $REGION \
  response.json

cat response.json
```

---

## Cost Breakdown

| Task | vCPU | Memory | Duration | Frequency | Cost |
|---|---|---|---|---|---|
| **Weekly scoring** | 0.25 | 512MB | 2 min | 4×/mo | $0.004 |
| **Monthly calibration** | 0.5 | 1GB | 5 min | 1×/mo | $0.009 |
| **Yearly retrain** | 1 | 2GB | 20 min | 1×/yr | $0.004/mo amortized |
| **S3 storage** | — | — | — | — | $0.50 |
| **CloudWatch logs** | — | — | — | — | $0.05 |
| **Lambda invocations** | — | — | — | — | Free (5 calls/mo) |
| **EventBridge rules** | — | — | — | — | Free |
| **ECR storage** | — | — | — | — | Free (first repo) |
| **Total/month** | — | — | — | — | **~$0.57** |

---

## Pricing Formula

**Fargate pricing:**
- vCPU-hours: $0.04645 per vCPU-hour
- Memory-hours: $0.00504 per GB-hour

**Weekly scoring (2 min, 0.25 vCPU, 512MB):**
```
= (0.25 vCPU × $0.04645) + (0.5 GB × $0.00504)
= $0.01161 + $0.00252
= $0.01413 per run × 4 runs/mo
= $0.0565/mo
```

**Monthly calibration (5 min, 0.5 vCPU, 1GB):**
```
= (0.5 vCPU × $0.04645) + (1 GB × $0.00504)
= $0.02323 + $0.00504
= $0.02827 per run × 1 run/mo
= $0.0283/mo
```

**Yearly retrain (20 min, 1 vCPU, 2GB):**
```
= (1 vCPU × $0.04645) + (2 GB × $0.00504)
= $0.04645 + $0.01008
= $0.05653 per run ÷ 12 months
= $0.0047/mo amortized
```

---

## Troubleshooting

### Task fails to start
- Check VPC/subnet/security group configuration
- Verify IAM role permissions
- Check ECR image exists and is accessible

### Task runs but exits quickly
- Check CloudWatch logs: `aws logs tail /ecs/forecast-score --follow`
- Common: S3 bucket not found, credentials missing, or script error

### High costs
- Reduce vCPU/memory if task completes early
- Check for stuck tasks: `aws ecs list-tasks --cluster forecast-cluster`

---

## Next Steps

1. Replace `subnet-12345678` and `sg-12345678` with your actual VPC resources
2. Test each Lambda function manually before relying on cron
3. Monitor CloudWatch logs for first 3 runs
4. Set up SNS alerts for task failures (optional)

