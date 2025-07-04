#!/usr/bin/env python3
"""
AWS Spot Instance Manager for GPU Workloads
Manages EC2 spot instances for cost-effective GPU compute
"""

import asyncio
import boto3
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class SpotInstanceManager:
    """Manages AWS EC2 spot instances for GPU workloads"""
    
    def __init__(self,
                 instance_type: str = "g4dn.xlarge",
                 max_price: float = 0.50,
                 region_name: str = "us-east-1",
                 key_name: Optional[str] = None,
                 security_group_ids: Optional[List[str]] = None,
                 subnet_id: Optional[str] = None,
                 user_data_script: Optional[str] = None):
        """
        Initialize spot instance manager
        
        Args:
            instance_type: EC2 instance type (e.g., g4dn.xlarge)
            max_price: Maximum spot price per hour
            region_name: AWS region
            key_name: EC2 key pair name for SSH access
            security_group_ids: List of security group IDs
            subnet_id: Subnet ID for instance placement
            user_data_script: User data script for instance initialization
        """
        self.instance_type = instance_type
        self.max_price = max_price
        self.region_name = region_name
        self.key_name = key_name
        self.security_group_ids = security_group_ids or []
        self.subnet_id = subnet_id
        self.user_data_script = user_data_script
        
        self.ec2_client = boto3.client('ec2', region_name=region_name)
        self.ec2_resource = boto3.resource('ec2', region_name=region_name)
        
        self.spot_request_id = None
        self.instance_id = None
        self.instance_ip = None
        
        logger.info(f"Initialized spot instance manager for {instance_type} in {region_name}")
        logger.info(f"Max spot price: ${max_price}/hour")
    
    async def get_spot_price_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get recent spot price history for the instance type
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            List of spot price data points
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            response = self.ec2_client.describe_spot_price_history(
                InstanceTypes=[self.instance_type],
                ProductDescriptions=['Linux/UNIX'],
                StartTime=start_time,
                EndTime=end_time,
                MaxResults=100
            )
            
            prices = response.get('SpotPrices', [])
            logger.info(f"Retrieved {len(prices)} spot price data points")
            
            if prices:
                current_price = float(prices[0]['SpotPrice'])
                avg_price = sum(float(p['SpotPrice']) for p in prices) / len(prices)
                min_price = min(float(p['SpotPrice']) for p in prices)
                max_price = max(float(p['SpotPrice']) for p in prices)
                
                logger.info(f"Spot price analysis for {self.instance_type}:")
                logger.info(f"  Current: ${current_price:.4f}/hr")
                logger.info(f"  Average: ${avg_price:.4f}/hr")
                logger.info(f"  Range: ${min_price:.4f} - ${max_price:.4f}/hr")
            
            return prices
            
        except Exception as e:
            logger.error(f"Error retrieving spot price history: {e}")
            return []
    
    def get_default_user_data(self) -> str:
        """
        Get default user data script for GPU instances
        
        Returns:
            User data script for instance initialization
        """
        return """#!/bin/bash
yum update -y

amazon-linux-extras install docker -y
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

amazon-linux-extras install python3.8 -y
python3 -m pip install --upgrade pip

python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python3 -m pip install transformers accelerate bitsandbytes

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

mkdir -p /home/ec2-user/fingpt-scoring
chown ec2-user:ec2-user /home/ec2-user/fingpt-scoring

/opt/aws/bin/cfn-signal -e $? --stack ${AWS::StackName} --resource SpotInstance --region ${AWS::Region}
"""
    
    async def find_optimal_availability_zone(self) -> Optional[str]:
        """
        Find the availability zone with the lowest current spot price
        
        Returns:
            Optimal availability zone or None if not found
        """
        try:
            response = self.ec2_client.describe_availability_zones()
            az_names = [az['ZoneName'] for az in response['AvailabilityZones']]
            
            response = self.ec2_client.describe_spot_price_history(
                InstanceTypes=[self.instance_type],
                ProductDescriptions=['Linux/UNIX'],
                MaxResults=len(az_names)
            )
            
            prices = response.get('SpotPrices', [])
            if not prices:
                logger.warning("No spot price data available")
                return az_names[0] if az_names else None
            
            az_prices = {}
            for price_data in prices:
                az = price_data['AvailabilityZone']
                price = float(price_data['SpotPrice'])
                if az not in az_prices or price < az_prices[az]:
                    az_prices[az] = price
            
            if az_prices:
                optimal_az = min(az_prices.keys(), key=lambda k: az_prices[k])
                optimal_price = az_prices[optimal_az]
                logger.info(f"Optimal AZ: {optimal_az} (${optimal_price:.4f}/hr)")
                return optimal_az
            
            return az_names[0] if az_names else None
            
        except Exception as e:
            logger.error(f"Error finding optimal availability zone: {e}")
            return None
    
    async def create_launch_template(self) -> Optional[str]:
        """
        Create or update launch template for spot instances
        
        Returns:
            Launch template ID or None if failed
        """
        try:
            template_name = f"fingpt-scoring-{self.instance_type}"
            
            response = self.ec2_client.describe_images(
                Owners=['amazon'],
                Filters=[
                    {'Name': 'name', 'Values': ['amzn2-ami-hvm-*-x86_64-gp2']},
                    {'Name': 'state', 'Values': ['available']},
                    {'Name': 'architecture', 'Values': ['x86_64']},
                ],
                MaxResults=1
            )
            
            if not response['Images']:
                logger.error("No suitable AMI found")
                return None
            
            ami_id = response['Images'][0]['ImageId']
            logger.info(f"Using AMI: {ami_id}")
            
            template_data = {
                'ImageId': ami_id,
                'InstanceType': self.instance_type,
                'UserData': self.user_data_script or self.get_default_user_data(),
                'IamInstanceProfile': {
                    'Name': 'EC2-S3-Access'  # Assume this role exists
                },
                'Monitoring': {'Enabled': True},
                'TagSpecifications': [
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': 'FinGPT-Scoring-Spot'},
                            {'Key': 'Purpose', 'Value': 'GPU-Sentiment-Analysis'},
                            {'Key': 'ManagedBy', 'Value': 'SpotInstanceManager'}
                        ]
                    }
                ]
            }
            
            if self.key_name:
                template_data['KeyName'] = self.key_name
            
            if self.security_group_ids:
                template_data['SecurityGroupIds'] = self.security_group_ids
            
            try:
                response = self.ec2_client.create_launch_template(
                    LaunchTemplateName=template_name,
                    LaunchTemplateData=template_data
                )
                template_id = response['LaunchTemplate']['LaunchTemplateId']
                logger.info(f"Created launch template: {template_id}")
                return template_id
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'InvalidLaunchTemplateName.AlreadyExistsException':
                    response = self.ec2_client.create_launch_template_version(
                        LaunchTemplateName=template_name,
                        LaunchTemplateData=template_data
                    )
                    template_id = response['LaunchTemplateVersion']['LaunchTemplateId']
                    logger.info(f"Updated launch template: {template_id}")
                    return template_id
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"Error creating launch template: {e}")
            return None
    
    async def launch_spot_instance(self) -> bool:
        """
        Launch a spot instance for GPU workloads
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Launching spot instance: {self.instance_type}")
            
            await self.get_spot_price_history()
            
            optimal_az = await self.find_optimal_availability_zone()
            
            template_id = await self.create_launch_template()
            if not template_id:
                logger.error("Failed to create launch template")
                return False
            
            spot_request = {
                'SpotPrice': str(self.max_price),
                'TargetCapacity': 1,
                'Type': 'request',
                'LaunchTemplateConfigs': [
                    {
                        'LaunchTemplateSpecification': {
                            'LaunchTemplateId': template_id,
                            'Version': '$Latest'
                        }
                    }
                ]
            }
            
            if optimal_az:
                spot_request['LaunchTemplateConfigs'][0]['Overrides'] = [
                    {'AvailabilityZone': optimal_az}
                ]
            
            response = self.ec2_client.request_spot_fleet(**spot_request)
            self.spot_request_id = response['SpotFleetRequestId']
            
            logger.info(f"Spot fleet request created: {self.spot_request_id}")
            
            success = await self._wait_for_spot_instance()
            
            if success:
                logger.info(f"✅ Spot instance launched successfully: {self.instance_id}")
                logger.info(f"   Instance IP: {self.instance_ip}")
                return True
            else:
                logger.error("❌ Failed to launch spot instance")
                return False
                
        except Exception as e:
            logger.error(f"Error launching spot instance: {e}")
            return False
    
    async def _wait_for_spot_instance(self, timeout: int = 600) -> bool:
        """
        Wait for spot instance to be running
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if instance is running, False if timeout or error
        """
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                response = self.ec2_client.describe_spot_fleet_requests(
                    SpotFleetRequestIds=[self.spot_request_id]
                )
                
                if not response['SpotFleetRequestConfigs']:
                    logger.error("Spot fleet request not found")
                    return False
                
                fleet_config = response['SpotFleetRequestConfigs'][0]
                state = fleet_config['SpotFleetRequestState']
                
                logger.info(f"Spot fleet state: {state}")
                
                if state == 'active':
                    response = self.ec2_client.describe_spot_fleet_instances(
                        SpotFleetRequestId=self.spot_request_id
                    )
                    
                    if response['ActiveInstances']:
                        instance = response['ActiveInstances'][0]
                        self.instance_id = instance['InstanceId']
                        
                        ec2_instance = self.ec2_resource.Instance(self.instance_id)
                        self.instance_ip = ec2_instance.public_ip_address
                        
                        ec2_instance.wait_until_running()
                        
                        return True
                
                elif state in ['cancelled_running', 'cancelled_terminating', 'failed']:
                    logger.error(f"Spot fleet request failed: {state}")
                    return False
                
                await asyncio.sleep(10)
            
            logger.error("Timeout waiting for spot instance")
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for spot instance: {e}")
            return False
    
    async def get_instance_status(self) -> Dict[str, Any]:
        """
        Get current status of managed spot instance
        
        Returns:
            Dict with instance status information
        """
        try:
            if not self.instance_id:
                return {'status': 'no_instance', 'message': 'No instance managed'}
            
            response = self.ec2_client.describe_instances(
                InstanceIds=[self.instance_id]
            )
            
            if not response['Reservations']:
                return {'status': 'not_found', 'message': 'Instance not found'}
            
            instance = response['Reservations'][0]['Instances'][0]
            state = instance['State']['Name']
            
            status_info = {
                'status': state,
                'instance_id': self.instance_id,
                'instance_type': instance['InstanceType'],
                'public_ip': instance.get('PublicIpAddress'),
                'private_ip': instance.get('PrivateIpAddress'),
                'launch_time': instance.get('LaunchTime'),
                'availability_zone': instance['Placement']['AvailabilityZone']
            }
            
            if state == 'running':
                spot_prices = await self.get_spot_price_history(days=1)
                if spot_prices:
                    current_price = float(spot_prices[0]['SpotPrice'])
                    status_info['current_spot_price'] = current_price
                    status_info['max_price'] = self.max_price
                    status_info['price_ok'] = current_price <= self.max_price
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting instance status: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def cleanup_spot_instance(self):
        """Clean up spot instance and associated resources"""
        try:
            if self.spot_request_id:
                logger.info(f"Cancelling spot fleet request: {self.spot_request_id}")
                
                self.ec2_client.cancel_spot_fleet_requests(
                    SpotFleetRequestIds=[self.spot_request_id],
                    TerminateInstances=True
                )
                
                await asyncio.sleep(30)
                
                self.spot_request_id = None
                self.instance_id = None
                self.instance_ip = None
                
                logger.info("✅ Spot instance cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up spot instance: {e}")
    
    async def estimate_cost(self, runtime_hours: float) -> Dict[str, Any]:
        """
        Estimate cost for running spot instance
        
        Args:
            runtime_hours: Expected runtime in hours
            
        Returns:
            Dict with cost estimates
        """
        try:
            spot_prices = await self.get_spot_price_history(days=7)
            
            if not spot_prices:
                return {
                    'estimated_cost': runtime_hours * self.max_price,
                    'max_cost': runtime_hours * self.max_price,
                    'avg_spot_price': self.max_price
                }
            
            current_price = float(spot_prices[0]['SpotPrice'])
            avg_price = sum(float(p['SpotPrice']) for p in spot_prices) / len(spot_prices)
            
            return {
                'estimated_cost': runtime_hours * current_price,
                'max_cost': runtime_hours * self.max_price,
                'avg_cost': runtime_hours * avg_price,
                'current_spot_price': current_price,
                'avg_spot_price': avg_price,
                'max_price': self.max_price,
                'runtime_hours': runtime_hours
            }
            
        except Exception as e:
            logger.error(f"Error estimating cost: {e}")
            return {
                'estimated_cost': runtime_hours * self.max_price,
                'max_cost': runtime_hours * self.max_price,
                'error': str(e)
            }


async def main():
    """Test spot instance manager functionality"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AWS Spot Instance Manager for GPU Workloads')
    parser.add_argument('--instance-type', default='g4dn.xlarge',
                        help='EC2 instance type')
    parser.add_argument('--max-price', type=float, default=0.50,
                        help='Maximum spot price per hour')
    parser.add_argument('--region', default='us-east-1',
                        help='AWS region')
    parser.add_argument('--action', choices=['launch', 'status', 'cleanup', 'prices'],
                        default='prices', help='Action to perform')
    parser.add_argument('--runtime-hours', type=float, default=1.0,
                        help='Expected runtime for cost estimation')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    manager = SpotInstanceManager(
        instance_type=args.instance_type,
        max_price=args.max_price,
        region_name=args.region
    )
    
    try:
        if args.action == 'prices':
            logger.info("📊 Getting spot price history...")
            await manager.get_spot_price_history()
            
            cost_estimate = await manager.estimate_cost(args.runtime_hours)
            logger.info(f"💰 Cost estimate for {args.runtime_hours} hours:")
            for key, value in cost_estimate.items():
                if isinstance(value, float):
                    logger.info(f"   {key}: ${value:.4f}")
                else:
                    logger.info(f"   {key}: {value}")
        
        elif args.action == 'launch':
            logger.info("🚀 Launching spot instance...")
            success = await manager.launch_spot_instance()
            if success:
                status = await manager.get_instance_status()
                logger.info(f"Instance status: {status}")
            
        elif args.action == 'status':
            logger.info("📋 Getting instance status...")
            status = await manager.get_instance_status()
            logger.info(f"Status: {json.dumps(status, indent=2, default=str)}")
        
        elif args.action == 'cleanup':
            logger.info("🧹 Cleaning up spot instance...")
            await manager.cleanup_spot_instance()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
