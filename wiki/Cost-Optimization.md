# Cost Optimization

ChemiAccel provides comprehensive cost optimization strategies for cloud computing, HPC usage, and resource management. This guide covers techniques to minimize computational costs while maintaining performance and reliability.

## Overview

Cost optimization features include:
- **Spot Instance Management**: Automated spot instance provisioning and management
- **Resource Right-Sizing**: Optimal instance type selection based on workload
- **Auto-Scaling**: Dynamic resource allocation to minimize idle costs
- **Multi-Cloud Optimization**: Cost-effective resource utilization across providers
- **Reserved Capacity Planning**: Strategic long-term resource reservations
- **Budget Management**: Real-time cost monitoring and alerting

## Quick Start

### Basic Cost Optimization

```python
from chemi_accel.cost import CostOptimizer, BudgetManager
from chemi_accel import Calculator

# Initialize cost optimizer
cost_optimizer = CostOptimizer(
    providers=["aws", "gcp", "azure"],
    budget_limit=1000.0,  # $1000 budget
    optimization_strategy="cost_performance_balanced"
)

# Configure budget monitoring
budget_manager = BudgetManager(
    daily_budget=50.0,
    monthly_budget=1000.0,
    alert_thresholds=[0.5, 0.8, 0.95]
)

# Optimize calculation with cost constraints
molecules = load_molecule_dataset()
optimized_config = cost_optimizer.optimize_for_budget(
    molecules=molecules,
    descriptors="comprehensive",
    max_cost=100.0,
    max_time_hours=6
)

print(f"Optimized configuration: ${optimized_config['estimated_cost']:.2f}")
print(f"Estimated completion time: {optimized_config['estimated_time']:.1f} hours")
```

## Cloud Cost Optimization

### Spot Instance Management

```python
from chemi_accel.cost.aws import SpotInstanceManager
from chemi_accel.cost.gcp import PreemptibleInstanceManager
from chemi_accel.cost.azure import SpotVMManager

class CloudSpotOptimizer:
    """Multi-cloud spot instance optimization for ChemiAccel."""
    
    def __init__(self, preferred_providers=None):
        self.preferred_providers = preferred_providers or ["aws", "gcp", "azure"]
        
        # Initialize spot managers for each provider
        self.spot_managers = {}
        if "aws" in self.preferred_providers:
            self.spot_managers["aws"] = SpotInstanceManager()
        if "gcp" in self.preferred_providers:
            self.spot_managers["gcp"] = PreemptibleInstanceManager()
        if "azure" in self.preferred_providers:
            self.spot_managers["azure"] = SpotVMManager()
    
    def get_current_spot_prices(self, instance_requirements):
        """Get current spot prices across all providers."""
        
        spot_prices = {}
        
        for provider, manager in self.spot_managers.items():
            try:
                prices = manager.get_current_spot_prices(
                    instance_types=instance_requirements['instance_types'],
                    regions=instance_requirements.get('regions', ['us-west-2', 'us-central1', 'eastus']),
                    availability_zones=instance_requirements.get('availability_zones')
                )
                spot_prices[provider] = prices
            except Exception as e:
                print(f"Failed to get spot prices from {provider}: {e}")
                spot_prices[provider] = {}
        
        return spot_prices
    
    def analyze_spot_price_history(self, instance_type, region, days=30):
        """Analyze spot price history to predict optimal bidding strategy."""
        
        historical_data = {}
        
        for provider, manager in self.spot_managers.items():
            try:
                history = manager.get_spot_price_history(
                    instance_type=instance_type,
                    region=region,
                    days=days
                )
                
                # Calculate statistics
                prices = [point['price'] for point in history]
                statistics = {
                    'mean_price': np.mean(prices),
                    'median_price': np.median(prices),
                    'min_price': np.min(prices),
                    'max_price': np.max(prices),
                    'std_dev': np.std(prices),
                    'percentile_90': np.percentile(prices, 90),
                    'interruption_rate': self.calculate_interruption_rate(history)
                }
                
                historical_data[provider] = {
                    'raw_data': history,
                    'statistics': statistics
                }
                
            except Exception as e:
                print(f"Failed to get historical data from {provider}: {e}")
        
        return historical_data
    
    def calculate_interruption_rate(self, price_history):
        """Calculate interruption rate from price history."""
        
        # Estimate interruptions based on price spikes
        prices = [point['price'] for point in price_history]
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        # Consider price spikes > 2 standard deviations as potential interruptions
        interruption_threshold = mean_price + 2 * std_price
        interruptions = sum(1 for price in prices if price > interruption_threshold)
        
        interruption_rate = interruptions / len(prices) if prices else 0
        return interruption_rate
    
    def optimize_spot_strategy(self, workload_config, cost_budget, fault_tolerance_level="medium"):
        """Optimize spot instance strategy for given workload."""
        
        instance_requirements = {
            'instance_types': workload_config['instance_types'],
            'regions': workload_config.get('regions', ['us-west-2', 'us-central1', 'eastus']),
            'min_instances': workload_config['min_instances'],
            'max_instances': workload_config['max_instances']
        }
        
        # Get current spot prices
        current_prices = self.get_current_spot_prices(instance_requirements)
        
        # Analyze historical data for pricing strategy
        historical_analysis = {}
        for instance_type in instance_requirements['instance_types']:
            for region in instance_requirements['regions']:
                key = f"{instance_type}_{region}"
                historical_analysis[key] = self.analyze_spot_price_history(
                    instance_type, region, days=30
                )
        
        # Calculate optimal bidding strategy
        optimization_strategy = self.calculate_optimal_bidding_strategy(
            current_prices=current_prices,
            historical_analysis=historical_analysis,
            cost_budget=cost_budget,
            fault_tolerance=fault_tolerance_level
        )
        
        return optimization_strategy
    
    def calculate_optimal_bidding_strategy(self, current_prices, historical_analysis, 
                                         cost_budget, fault_tolerance):
        """Calculate optimal bidding strategy across providers."""
        
        strategies = []
        
        for provider, prices in current_prices.items():
            for region, region_prices in prices.items():
                for instance_type, current_price in region_prices.items():
                    
                    # Get historical statistics
                    key = f"{instance_type}_{region}"
                    if key in historical_analysis and provider in historical_analysis[key]:
                        stats = historical_analysis[key][provider]['statistics']
                        
                        # Calculate bid price based on fault tolerance level
                        if fault_tolerance == "low":
                            # Bid high to minimize interruptions
                            bid_price = stats['percentile_90']
                            expected_interruption_rate = stats['interruption_rate'] * 0.3
                        elif fault_tolerance == "medium":
                            # Balanced approach
                            bid_price = stats['median_price'] * 1.2
                            expected_interruption_rate = stats['interruption_rate'] * 0.6
                        else:  # high fault tolerance
                            # Bid low, accept more interruptions
                            bid_price = stats['median_price']
                            expected_interruption_rate = stats['interruption_rate']
                        
                        # Calculate cost-effectiveness score
                        cost_per_hour = min(bid_price, current_price)
                        effectiveness_score = (1 - expected_interruption_rate) / cost_per_hour
                        
                        strategy = {
                            'provider': provider,
                            'region': region,
                            'instance_type': instance_type,
                            'bid_price': bid_price,
                            'current_price': current_price,
                            'expected_interruption_rate': expected_interruption_rate,
                            'cost_effectiveness_score': effectiveness_score,
                            'recommended_allocation': 0  # To be calculated
                        }
                        
                        strategies.append(strategy)
        
        # Sort by cost-effectiveness and allocate budget
        strategies.sort(key=lambda x: x['cost_effectiveness_score'], reverse=True)
        
        # Allocate instances based on budget and cost-effectiveness
        remaining_budget = cost_budget
        total_instances_needed = 10  # Example, would be calculated from workload
        
        for strategy in strategies:
            if remaining_budget <= 0:
                break
            
            max_instances_affordable = int(remaining_budget / strategy['bid_price'])
            instances_to_allocate = min(max_instances_affordable, 
                                      max(1, total_instances_needed // len(strategies)))
            
            strategy['recommended_allocation'] = instances_to_allocate
            remaining_budget -= instances_to_allocate * strategy['bid_price']
            total_instances_needed -= instances_to_allocate
        
        return strategies
    
    def implement_diversified_spot_strategy(self, optimization_strategies):
        """Implement diversified spot instance strategy across providers."""
        
        active_instances = {}
        
        for strategy in optimization_strategies:
            if strategy['recommended_allocation'] > 0:
                provider = strategy['provider']
                manager = self.spot_managers[provider]
                
                # Request spot instances
                try:
                    instances = manager.request_spot_instances(
                        instance_type=strategy['instance_type'],
                        region=strategy['region'],
                        count=strategy['recommended_allocation'],
                        bid_price=strategy['bid_price'],
                        user_data=self.generate_chemiaccel_user_data()
                    )
                    
                    active_instances[f"{provider}_{strategy['region']}"] = {
                        'instances': instances,
                        'strategy': strategy
                    }
                    
                    print(f"Requested {strategy['recommended_allocation']} {strategy['instance_type']} "
                          f"instances from {provider} in {strategy['region']} at ${strategy['bid_price']:.4f}/hour")
                
                except Exception as e:
                    print(f"Failed to request instances from {provider}: {e}")
        
        return active_instances
    
    def generate_chemiaccel_user_data(self):
        """Generate user data script for ChemiAccel spot instances."""
        
        user_data = '''#!/bin/bash
# Install ChemiAccel and dependencies
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Pull ChemiAccel container
docker pull chemiaccel/worker:latest

# Set up spot instance interrupt handler
cat > /opt/spot-interrupt-handler.sh << EOF
#!/bin/bash
# Check for spot interruption notice
TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`
INTERRUPTION=`curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/spot/instance-action`

if [ "$INTERRUPTION" != "" ]; then
    echo "Spot interruption detected, saving checkpoint..."
    docker exec chemiaccel-worker python -c "
    import signal
    import os
    os.kill(1, signal.SIGUSR1)  # Send checkpoint signal
    "
    sleep 30  # Give time for checkpoint
    docker stop chemiaccel-worker
fi
EOF

chmod +x /opt/spot-interrupt-handler.sh

# Set up cron job to check for interruptions
echo "* * * * * /opt/spot-interrupt-handler.sh" | crontab -

# Start ChemiAccel worker
docker run -d --name chemiaccel-worker \
    -e MASTER_HOST=${MASTER_HOST} \
    -e WORKER_ID=$(cat /proc/sys/kernel/random/uuid) \
    chemiaccel/worker:latest
'''
        
        return user_data

# Example: Comprehensive spot optimization
spot_optimizer = CloudSpotOptimizer(preferred_providers=["aws", "gcp", "azure"])

# Define workload requirements
workload_config = {
    'instance_types': ['g4dn.xlarge', 'n1-standard-4-k80', 'Standard_NC6s_v3'],
    'regions': ['us-west-2', 'us-central1', 'eastus'],
    'min_instances': 5,
    'max_instances': 50
}

# Optimize spot strategy with $200 budget
optimization_strategies = spot_optimizer.optimize_spot_strategy(
    workload_config=workload_config,
    cost_budget=200.0,
    fault_tolerance_level="medium"
)

print("Optimal spot strategies:")
for strategy in optimization_strategies[:5]:  # Show top 5
    print(f"  {strategy['provider']} {strategy['instance_type']} in {strategy['region']}: "
          f"${strategy['bid_price']:.4f}/hr, {strategy['recommended_allocation']} instances")

# Implement diversified strategy
active_instances = spot_optimizer.implement_diversified_spot_strategy(optimization_strategies)
```

### Reserved Instance Optimization

```python
from chemi_accel.cost import ReservedInstanceOptimizer

class ReservedCapacityPlanner:
    """Plan and optimize reserved instance purchases."""
    
    def __init__(self):
        self.ri_optimizer = ReservedInstanceOptimizer()
        self.usage_analyzer = UsageAnalyzer()
    
    def analyze_historical_usage(self, months=12):
        """Analyze historical usage patterns for RI recommendations."""
        
        usage_data = self.usage_analyzer.get_usage_history(months=months)
        
        analysis = {
            'instance_types': {},
            'regions': {},
            'usage_patterns': {}
        }
        
        for usage_record in usage_data:
            instance_type = usage_record['instance_type']
            region = usage_record['region']
            hours_used = usage_record['hours_used']
            
            # Track instance type usage
            if instance_type not in analysis['instance_types']:
                analysis['instance_types'][instance_type] = {
                    'total_hours': 0,
                    'monthly_avg': 0,
                    'peak_usage': 0,
                    'baseline_usage': 0
                }
            
            analysis['instance_types'][instance_type]['total_hours'] += hours_used
            analysis['instance_types'][instance_type]['peak_usage'] = max(
                analysis['instance_types'][instance_type]['peak_usage'],
                hours_used
            )
            
            # Track regional usage
            if region not in analysis['regions']:
                analysis['regions'][region] = {'total_hours': 0}
            analysis['regions'][region]['total_hours'] += hours_used
        
        # Calculate averages and baseline usage
        for instance_type, data in analysis['instance_types'].items():
            data['monthly_avg'] = data['total_hours'] / months
            data['baseline_usage'] = data['monthly_avg'] * 0.8  # 80% of average as baseline
        
        return analysis
    
    def calculate_ri_recommendations(self, usage_analysis, cost_optimization_target=0.3):
        """Calculate reserved instance recommendations."""
        
        recommendations = []
        
        for instance_type, usage_data in usage_analysis['instance_types'].items():
            baseline_hours = usage_data['baseline_usage']
            
            if baseline_hours >= 100:  # Only recommend if significant usage
                
                # Calculate potential savings
                on_demand_rate = self.get_on_demand_rate(instance_type)
                ri_rates = self.get_ri_rates(instance_type)
                
                for term, ri_rate in ri_rates.items():
                    monthly_on_demand_cost = baseline_hours * on_demand_rate
                    monthly_ri_cost = baseline_hours * ri_rate['hourly'] + ri_rate['upfront'] / (int(term) * 12)
                    
                    monthly_savings = monthly_on_demand_cost - monthly_ri_cost
                    savings_percentage = monthly_savings / monthly_on_demand_cost
                    
                    if savings_percentage >= cost_optimization_target:
                        recommendation = {
                            'instance_type': instance_type,
                            'term': term,
                            'payment_option': ri_rate['payment_option'],
                            'recommended_quantity': int(baseline_hours / 730),  # Convert to instances
                            'monthly_savings': monthly_savings,
                            'savings_percentage': savings_percentage,
                            'upfront_cost': ri_rate['upfront'],
                            'payback_period_months': ri_rate['upfront'] / monthly_savings if monthly_savings > 0 else float('inf')
                        }
                        recommendations.append(recommendation)
        
        # Sort by savings potential
        recommendations.sort(key=lambda x: x['monthly_savings'], reverse=True)
        
        return recommendations
    
    def create_ri_purchase_plan(self, recommendations, budget_limit, risk_tolerance="medium"):
        """Create optimized RI purchase plan within budget."""
        
        purchase_plan = []
        remaining_budget = budget_limit
        total_monthly_savings = 0
        
        # Filter recommendations based on risk tolerance
        filtered_recommendations = self.filter_by_risk_tolerance(recommendations, risk_tolerance)
        
        for rec in filtered_recommendations:
            upfront_cost = rec['upfront_cost'] * rec['recommended_quantity']
            
            if upfront_cost <= remaining_budget and rec['payback_period_months'] <= 12:
                purchase_plan.append(rec)
                remaining_budget -= upfront_cost
                total_monthly_savings += rec['monthly_savings']
        
        plan_summary = {
            'purchases': purchase_plan,
            'total_upfront_cost': budget_limit - remaining_budget,
            'total_monthly_savings': total_monthly_savings,
            'annual_savings': total_monthly_savings * 12,
            'roi_percentage': (total_monthly_savings * 12) / (budget_limit - remaining_budget) if remaining_budget < budget_limit else 0
        }
        
        return plan_summary
    
    def filter_by_risk_tolerance(self, recommendations, risk_tolerance):
        """Filter recommendations based on risk tolerance."""
        
        if risk_tolerance == "low":
            # Conservative: only 1-year terms, no partial upfront
            return [r for r in recommendations 
                   if r['term'] == '1year' and r['payment_option'] in ['no_upfront', 'partial_upfront']]
        
        elif risk_tolerance == "medium":
            # Balanced: 1-year and 3-year terms
            return [r for r in recommendations 
                   if r['payback_period_months'] <= 18]
        
        else:  # high risk tolerance
            # Aggressive: include all terms and payment options
            return recommendations

# Example: Reserved instance optimization
ri_planner = ReservedCapacityPlanner()

# Analyze historical usage
usage_analysis = ri_planner.analyze_historical_usage(months=12)

print("Historical usage analysis:")
for instance_type, data in usage_analysis['instance_types'].items():
    print(f"  {instance_type}: {data['monthly_avg']:.0f} hours/month average, "
          f"{data['baseline_usage']:.0f} hours baseline")

# Get RI recommendations
ri_recommendations = ri_planner.calculate_ri_recommendations(
    usage_analysis=usage_analysis,
    cost_optimization_target=0.25  # Target 25% savings
)

print(f"\nFound {len(ri_recommendations)} RI recommendations")

# Create purchase plan with $10,000 budget
purchase_plan = ri_planner.create_ri_purchase_plan(
    recommendations=ri_recommendations,
    budget_limit=10000.0,
    risk_tolerance="medium"
)

print(f"\nRecommended RI purchase plan:")
print(f"  Total upfront cost: ${purchase_plan['total_upfront_cost']:.2f}")
print(f"  Monthly savings: ${purchase_plan['total_monthly_savings']:.2f}")
print(f"  Annual ROI: {purchase_plan['roi_percentage']:.1%}")

for purchase in purchase_plan['purchases'][:5]:  # Show top 5
    print(f"  Buy {purchase['recommended_quantity']}x {purchase['instance_type']} "
          f"({purchase['term']}) - saves ${purchase['monthly_savings']:.2f}/month")
```

## Budget Management and Monitoring

### Real-time Cost Tracking

```python
from chemi_accel.cost import CostTracker, BudgetAlertManager

class RealTimeCostMonitor:
    """Real-time cost monitoring and budget management."""
    
    def __init__(self, budget_config):
        self.budget_config = budget_config
        self.cost_tracker = CostTracker()
        self.alert_manager = BudgetAlertManager()
        self.current_costs = {}
        self.cost_history = []
        
    def initialize_cost_tracking(self):
        """Initialize cost tracking across all providers."""
        
        tracking_config = {
            'providers': self.budget_config['providers'],
            'update_interval': 300,  # 5 minutes
            'cost_categories': [
                'compute', 'storage', 'network', 'gpu', 'spot_instances'
            ],
            'granularity': 'service'  # Track costs by service
        }
        
        self.cost_tracker.configure(tracking_config)
        
        # Set up budget alerts
        for alert_config in self.budget_config['alerts']:
            self.alert_manager.add_alert(
                threshold=alert_config['threshold'],
                threshold_type=alert_config['type'],  # 'percentage' or 'absolute'
                notification_method=alert_config['notification_method'],
                alert_frequency=alert_config.get('frequency', 'once')
            )
    
    def get_current_costs(self):
        """Get current costs across all providers and services."""
        
        current_costs = {}
        
        for provider in self.budget_config['providers']:
            try:
                provider_costs = self.cost_tracker.get_provider_costs(
                    provider=provider,
                    time_period='current_month'
                )
                current_costs[provider] = provider_costs
            except Exception as e:
                print(f"Failed to get costs from {provider}: {e}")
                current_costs[provider] = {'total': 0, 'services': {}}
        
        # Calculate totals
        total_cost = sum(provider['total'] for provider in current_costs.values())
        
        cost_summary = {
            'total_cost': total_cost,
            'provider_costs': current_costs,
            'timestamp': time.time()
        }
        
        self.current_costs = cost_summary
        self.cost_history.append(cost_summary)
        
        # Keep only recent history (last 24 hours)
        cutoff_time = time.time() - 86400
        self.cost_history = [c for c in self.cost_history if c['timestamp'] > cutoff_time]
        
        return cost_summary
    
    def check_budget_alerts(self):
        """Check budget thresholds and trigger alerts if needed."""
        
        current_costs = self.get_current_costs()
        
        # Check against daily budget
        if 'daily_budget' in self.budget_config:
            daily_costs = self.calculate_daily_costs()
            if daily_costs > self.budget_config['daily_budget']:
                self.alert_manager.trigger_alert(
                    alert_type='daily_budget_exceeded',
                    current_amount=daily_costs,
                    budget_amount=self.budget_config['daily_budget'],
                    message=f"Daily budget exceeded: ${daily_costs:.2f} > ${self.budget_config['daily_budget']:.2f}"
                )
        
        # Check against monthly budget
        if 'monthly_budget' in self.budget_config:
            monthly_costs = current_costs['total_cost']
            budget_percentage = monthly_costs / self.budget_config['monthly_budget']
            
            # Check percentage thresholds
            for threshold in [0.5, 0.75, 0.9, 1.0]:
                if budget_percentage >= threshold:
                    self.alert_manager.trigger_alert(
                        alert_type='monthly_budget_threshold',
                        current_amount=monthly_costs,
                        budget_amount=self.budget_config['monthly_budget'],
                        threshold_percentage=threshold,
                        message=f"Monthly budget {threshold:.0%} threshold reached: "
                               f"${monthly_costs:.2f} / ${self.budget_config['monthly_budget']:.2f}"
                    )
        
        # Check cost anomalies
        self.check_cost_anomalies()
    
    def calculate_daily_costs(self):
        """Calculate costs for the current day."""
        
        if len(self.cost_history) < 2:
            return 0
        
        # Get costs from 24 hours ago and now
        current_time = time.time()
        day_start = current_time - 86400
        
        # Find cost record closest to 24 hours ago
        start_costs = min(self.cost_history, 
                         key=lambda c: abs(c['timestamp'] - day_start))
        current_costs = self.cost_history[-1]
        
        daily_cost = current_costs['total_cost'] - start_costs['total_cost']
        return max(0, daily_cost)  # Ensure non-negative
    
    def check_cost_anomalies(self):
        """Check for unusual cost spikes or patterns."""
        
        if len(self.cost_history) < 10:  # Need sufficient data
            return
        
        recent_costs = [c['total_cost'] for c in self.cost_history[-10:]]
        
        # Calculate moving average and standard deviation
        moving_avg = np.mean(recent_costs[:-1])  # Exclude current
        std_dev = np.std(recent_costs[:-1])
        current_cost = recent_costs[-1]
        
        # Check for anomalies (cost spike > 2 standard deviations)
        if std_dev > 0 and abs(current_cost - moving_avg) > 2 * std_dev:
            if current_cost > moving_avg:
                anomaly_type = 'cost_spike'
                message = f"Cost spike detected: ${current_cost:.2f} (avg: ${moving_avg:.2f})"
            else:
                anomaly_type = 'cost_drop'
                message = f"Unusual cost drop: ${current_cost:.2f} (avg: ${moving_avg:.2f})"
            
            self.alert_manager.trigger_alert(
                alert_type=anomaly_type,
                current_amount=current_cost,
                expected_amount=moving_avg,
                message=message
            )
    
    def generate_cost_forecast(self, forecast_days=30):
        """Generate cost forecast based on historical trends."""
        
        if len(self.cost_history) < 7:  # Need at least a week of data
            return None
        
        # Extract daily cost trends
        daily_costs = []
        for i in range(1, len(self.cost_history)):
            daily_cost = (self.cost_history[i]['total_cost'] - 
                         self.cost_history[i-1]['total_cost'])
            daily_costs.append(max(0, daily_cost))
        
        # Simple linear trend forecast
        days = np.arange(len(daily_costs))
        trend = np.polyfit(days, daily_costs, 1)
        
        # Forecast future costs
        forecast = []
        base_cost = self.cost_history[-1]['total_cost']
        
        for day in range(1, forecast_days + 1):
            daily_trend_cost = np.polyval(trend, len(daily_costs) + day - 1)
            forecast_cost = base_cost + daily_trend_cost * day
            forecast.append({
                'day': day,
                'forecasted_total': forecast_cost,
                'daily_cost': daily_trend_cost
            })
        
        return {
            'forecast': forecast,
            'current_monthly_total': base_cost,
            'forecasted_monthly_total': forecast[-1]['forecasted_total'] if forecast else base_cost,
            'trend_daily_cost': np.polyval(trend, len(daily_costs))
        }
    
    def implement_cost_controls(self, control_config):
        """Implement automated cost controls."""
        
        controls = []
        
        # Auto-scaling controls
        if 'auto_scaling' in control_config:
            scaling_control = {
                'type': 'auto_scaling',
                'max_instances': control_config['auto_scaling']['max_instances'],
                'cost_threshold': control_config['auto_scaling']['cost_threshold'],
                'scale_down_aggressive': control_config['auto_scaling'].get('aggressive_scale_down', False)
            }
            controls.append(scaling_control)
        
        # Spot instance controls
        if 'spot_instances' in control_config:
            spot_control = {
                'type': 'spot_management',
                'max_bid_price': control_config['spot_instances']['max_bid_price'],
                'diversification_required': control_config['spot_instances'].get('diversify', True),
                'fallback_to_on_demand': control_config['spot_instances'].get('fallback', True)
            }
            controls.append(spot_control)
        
        # Resource right-sizing controls
        if 'right_sizing' in control_config:
            sizing_control = {
                'type': 'right_sizing',
                'utilization_threshold': control_config['right_sizing']['utilization_threshold'],
                'resize_frequency': control_config['right_sizing'].get('check_frequency', 'hourly'),
                'allowed_downgrades': control_config['right_sizing'].get('allow_downgrades', True)
            }
            controls.append(sizing_control)
        
        return self.cost_tracker.implement_controls(controls)

# Example: Real-time cost monitoring setup
budget_config = {
    'providers': ['aws', 'gcp', 'azure'],
    'daily_budget': 100.0,
    'monthly_budget': 2000.0,
    'alerts': [
        {
            'threshold': 0.8,
            'type': 'percentage',
            'notification_method': 'email',
            'frequency': 'once_per_day'
        },
        {
            'threshold': 150.0,
            'type': 'absolute',
            'notification_method': 'slack',
            'frequency': 'immediate'
        }
    ]
}

cost_monitor = RealTimeCostMonitor(budget_config)
cost_monitor.initialize_cost_tracking()

# Monitor costs continuously
def continuous_cost_monitoring():
    while True:
        try:
            # Get current costs
            current_costs = cost_monitor.get_current_costs()
            print(f"Current total cost: ${current_costs['total_cost']:.2f}")
            
            # Check budget alerts
            cost_monitor.check_budget_alerts()
            
            # Generate forecast
            forecast = cost_monitor.generate_cost_forecast(forecast_days=30)
            if forecast:
                print(f"Forecasted monthly total: ${forecast['forecasted_monthly_total']:.2f}")
                
                # Check if forecast exceeds budget
                if forecast['forecasted_monthly_total'] > budget_config['monthly_budget']:
                    print("WARNING: Forecasted costs exceed monthly budget!")
                    
                    # Implement cost controls
                    control_config = {
                        'auto_scaling': {
                            'max_instances': 20,
                            'cost_threshold': budget_config['monthly_budget'] * 0.9
                        },
                        'spot_instances': {
                            'max_bid_price': 0.5,
                            'diversify': True
                        }
                    }
                    
                    cost_monitor.implement_cost_controls(control_config)
            
            time.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            print(f"Cost monitoring error: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

# Start monitoring (in production, run as background service)
# continuous_cost_monitoring()
```

## Resource Right-Sizing

### Intelligent Instance Selection

```python
from chemi_accel.cost import ResourceOptimizer, PerformanceBenchmarker

class IntelligentResourceSizer:
    """Intelligent resource sizing based on workload characteristics."""
    
    def __init__(self):
        self.resource_optimizer = ResourceOptimizer()
        self.benchmarker = PerformanceBenchmarker()
        self.performance_cache = {}
    
    def analyze_workload_requirements(self, molecules, descriptors):
        """Analyze workload to determine resource requirements."""
        
        # Sample workload for analysis
        sample_molecules = molecules[:min(1000, len(molecules))]
        
        workload_analysis = {
            'molecule_count': len(molecules),
            'avg_molecule_size': np.mean([len(mol) for mol in sample_molecules]),
            'descriptor_complexity': self.estimate_descriptor_complexity(descriptors),
            'memory_requirement': self.estimate_memory_requirement(sample_molecules, descriptors),
            'compute_intensity': self.estimate_compute_intensity(sample_molecules, descriptors),
            'io_requirements': self.estimate_io_requirements(len(molecules))
        }
        
        return workload_analysis
    
    def benchmark_instance_types(self, workload_analysis, candidate_instances):
        """Benchmark different instance types for the workload."""
        
        benchmark_results = {}
        
        for instance_type in candidate_instances:
            # Check cache first
            cache_key = f"{instance_type}_{workload_analysis['descriptor_complexity']}"
            
            if cache_key in self.performance_cache:
                benchmark_results[instance_type] = self.performance_cache[cache_key]
                continue
            
            try:
                # Run benchmark
                benchmark_result = self.benchmarker.run_instance_benchmark(
                    instance_type=instance_type,
                    workload_profile=workload_analysis,
                    duration_minutes=5
                )
                
                benchmark_results[instance_type] = benchmark_result
                self.performance_cache[cache_key] = benchmark_result
                
            except Exception as e:
                print(f"Benchmark failed for {instance_type}: {e}")
                benchmark_results[instance_type] = None
        
        return benchmark_results
    
    def calculate_cost_performance_ratio(self, benchmark_results):
        """Calculate cost-performance ratio for each instance type."""
        
        cost_performance_ratios = {}
        
        for instance_type, benchmark in benchmark_results.items():
            if benchmark is None:
                continue
            
            # Get pricing information
            pricing = self.resource_optimizer.get_instance_pricing(instance_type)
            
            # Calculate performance score
            performance_score = (
                benchmark['throughput'] * 1.0 +           # molecules/second
                (1 / benchmark['latency']) * 0.5 +        # inverse latency
                benchmark['resource_efficiency'] * 0.3    # resource utilization
            )
            
            # Calculate cost per performance unit
            cost_per_performance = pricing['hourly_rate'] / performance_score
            
            cost_performance_ratios[instance_type] = {
                'cost_per_hour': pricing['hourly_rate'],
                'performance_score': performance_score,
                'cost_per_performance': cost_per_performance,
                'throughput': benchmark['throughput'],
                'memory_efficiency': benchmark['memory_efficiency'],
                'benchmark_details': benchmark
            }
        
        return cost_performance_ratios
    
    def recommend_optimal_instances(self, workload_analysis, budget_constraints=None):
        """Recommend optimal instance types based on workload and budget."""
        
        # Get candidate instance types based on workload requirements
        candidate_instances = self.resource_optimizer.get_candidate_instances(
            memory_gb=workload_analysis['memory_requirement'],
            cpu_cores=self.estimate_cpu_requirements(workload_analysis),
            gpu_required=workload_analysis['compute_intensity'] > 0.7,
            network_performance=workload_analysis['io_requirements']
        )
        
        # Benchmark candidate instances
        benchmark_results = self.benchmark_instance_types(workload_analysis, candidate_instances)
        
        # Calculate cost-performance ratios
        cost_performance_ratios = self.calculate_cost_performance_ratio(benchmark_results)
        
        # Sort by cost-performance ratio (lower is better)
        sorted_recommendations = sorted(
            cost_performance_ratios.items(),
            key=lambda x: x[1]['cost_per_performance']
        )
        
        recommendations = []
        for instance_type, metrics in sorted_recommendations:
            
            # Calculate estimated total cost and time
            estimated_time_hours = workload_analysis['molecule_count'] / (metrics['throughput'] * 3600)
            estimated_total_cost = estimated_time_hours * metrics['cost_per_hour']
            
            # Check budget constraints
            within_budget = True
            if budget_constraints:
                if 'max_total_cost' in budget_constraints:
                    within_budget = estimated_total_cost <= budget_constraints['max_total_cost']
                if 'max_hourly_cost' in budget_constraints and within_budget:
                    within_budget = metrics['cost_per_hour'] <= budget_constraints['max_hourly_cost']
            
            recommendation = {
                'instance_type': instance_type,
                'cost_per_hour': metrics['cost_per_hour'],
                'estimated_time_hours': estimated_time_hours,
                'estimated_total_cost': estimated_total_cost,
                'performance_score': metrics['performance_score'],
                'cost_efficiency_rank': len(recommendations) + 1,
                'within_budget': within_budget,
                'pros': self.identify_instance_pros(instance_type, metrics),
                'cons': self.identify_instance_cons(instance_type, metrics)
            }
            
            recommendations.append(recommendation)
        
        return recommendations[:10]  # Return top 10 recommendations
    
    def identify_instance_pros(self, instance_type, metrics):
        """Identify pros for an instance type."""
        
        pros = []
        
        if metrics['cost_per_performance'] < 0.1:
            pros.append("Excellent cost-performance ratio")
        
        if metrics['throughput'] > 100:
            pros.append("High throughput performance")
        
        if metrics['memory_efficiency'] > 0.8:
            pros.append("Efficient memory utilization")
        
        if 'gpu' in instance_type.lower():
            pros.append("GPU acceleration available")
        
        if metrics['cost_per_hour'] < 1.0:
            pros.append("Low hourly cost")
        
        return pros
    
    def identify_instance_cons(self, instance_type, metrics):
        """Identify cons for an instance type."""
        
        cons = []
        
        if metrics['cost_per_performance'] > 0.5:
            cons.append("High cost-performance ratio")
        
        if metrics['throughput'] < 10:
            cons.append("Low throughput performance")
        
        if metrics['memory_efficiency'] < 0.5:
            cons.append("Poor memory utilization")
        
        if metrics['cost_per_hour'] > 5.0:
            cons.append("High hourly cost")
        
        return cons
    
    def create_scaling_strategy(self, recommendations, workload_size):
        """Create auto-scaling strategy based on recommendations."""
        
        # Select top 3 instance types for diversification
        top_instances = recommendations[:3]
        
        scaling_strategy = {
            'primary_instance': top_instances[0]['instance_type'],
            'secondary_instances': [rec['instance_type'] for rec in top_instances[1:3]],
            'scaling_rules': []
        }
        
        # Define scaling rules based on workload size
        if workload_size < 10000:  # Small workload
            scaling_strategy['scaling_rules'] = [
                {'condition': 'queue_length > 100', 'action': 'scale_up', 'instances': 1},
                {'condition': 'queue_length < 10', 'action': 'scale_down', 'instances': 1},
                {'condition': 'cost_per_hour > 5.0', 'action': 'switch_to_secondary'}
            ]
        
        elif workload_size < 100000:  # Medium workload
            scaling_strategy['scaling_rules'] = [
                {'condition': 'queue_length > 500', 'action': 'scale_up', 'instances': 2},
                {'condition': 'queue_length < 50', 'action': 'scale_down', 'instances': 1},
                {'condition': 'avg_utilization < 30%', 'action': 'downgrade_instance_type'}
            ]
        
        else:  # Large workload
            scaling_strategy['scaling_rules'] = [
                {'condition': 'queue_length > 1000', 'action': 'scale_up', 'instances': 5},
                {'condition': 'processing_rate < target_rate * 0.8', 'action': 'upgrade_instance_type'},
                {'condition': 'cost_trend > budget_limit', 'action': 'switch_to_spot_instances'}
            ]
        
        return scaling_strategy

# Example: Intelligent resource sizing
resource_sizer = IntelligentResourceSizer()

# Analyze workload
molecules = load_large_dataset()  # 500K molecules
workload_analysis = resource_sizer.analyze_workload_requirements(
    molecules=molecules,
    descriptors="comprehensive"
)

print("Workload Analysis:")
print(f"  Molecules: {workload_analysis['molecule_count']:,}")
print(f"  Memory requirement: {workload_analysis['memory_requirement']:.1f} GB")
print(f"  Compute intensity: {workload_analysis['compute_intensity']:.2f}")

# Get recommendations with budget constraints
budget_constraints = {
    'max_total_cost': 500.0,    # $500 total budget
    'max_hourly_cost': 10.0     # $10/hour max
}

recommendations = resource_sizer.recommend_optimal_instances(
    workload_analysis=workload_analysis,
    budget_constraints=budget_constraints
)

print(f"\nTop 5 Instance Recommendations:")
for i, rec in enumerate(recommendations[:5]):
    status = "✓ Within budget" if rec['within_budget'] else "✗ Over budget"
    print(f"{i+1}. {rec['instance_type']}")
    print(f"   Cost: ${rec['cost_per_hour']:.2f}/hr (${rec['estimated_total_cost']:.2f} total)")
    print(f"   Time: {rec['estimated_time_hours']:.1f} hours")
    print(f"   Status: {status}")
    print(f"   Pros: {', '.join(rec['pros'][:2])}")

# Create scaling strategy
scaling_strategy = resource_sizer.create_scaling_strategy(
    recommendations=recommendations,
    workload_size=len(molecules)
)

print(f"\nRecommended Scaling Strategy:")
print(f"  Primary instance: {scaling_strategy['primary_instance']}")
print(f"  Secondary instances: {scaling_strategy['secondary_instances']}")
print(f"  Scaling rules: {len(scaling_strategy['scaling_rules'])} configured")
```

## Best Practices

### 1. Cost Monitoring Setup
```python
# Set up comprehensive cost monitoring
def setup_cost_monitoring():
    monitoring_config = {
        'update_frequency': 300,  # 5 minutes
        'alert_channels': ['email', 'slack', 'webhook'],
        'cost_breakdown': ['service', 'region', 'instance_type'],
        'budget_tracking': ['daily', 'weekly', 'monthly'],
        'anomaly_detection': True
    }
    return monitoring_config
```

### 2. Multi-Cloud Cost Optimization
```python
# Optimize costs across multiple cloud providers
def optimize_multi_cloud_costs():
    optimization_strategy = {
        'provider_selection': 'cost_performance_balanced',
        'workload_distribution': 'intelligent',
        'spot_instance_diversification': True,
        'reserved_capacity_planning': True,
        'real_time_cost_comparison': True
    }
    return optimization_strategy
```

### 3. Automated Cost Controls
```python
# Implement automated cost controls
def setup_automated_controls():
    controls = {
        'budget_limits': {'daily': 100, 'monthly': 2000},
        'auto_scaling': {'max_instances': 50, 'cost_threshold': 1800},
        'spot_management': {'max_interruption_rate': 0.1},
        'resource_right_sizing': {'utilization_threshold': 0.7},
        'emergency_shutdown': {'cost_spike_threshold': 2.0}
    }
    return controls
```

---

*For specific cloud provider cost optimization, see the [Cloud Deployment](Cloud-Deployment) guide.*
