#!/usr/bin/env python3
"""
Reinforcement Learning Agent for AI Assistant Feedback Loop

This module implements a simple RL agent that adjusts the AI assistant's behavior
based on user feedback (thumbs up/down).
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLAgent:
    """Simple RL agent for adjusting AI assistant behavior based on feedback."""
    
    def __init__(self, db_path: str = "assistant_demo.db"):
        self.db_path = db_path
        # Initialize weights for different components
        self.weights = {
            'summarization_quality': 0.5,
            'task_relevance': 0.5,
            'response_helpfulness': 0.5
        }
        self.learning_rate = 0.1
        self.feedback_history = []
        
    def collect_feedback(self) -> List[Dict]:
        """Collect recent feedback from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent feedback (last 24 hours or last 100 entries)
            cursor.execute('''
                SELECT summary_id, task_id, response_id, score, comment, timestamp
                FROM coach_feedback
                ORDER BY timestamp DESC
                LIMIT 100
            ''')
            
            feedback_records = cursor.fetchall()
            conn.close()
            
            feedback_list = []
            for record in feedback_records:
                feedback_list.append({
                    'summary_id': record[0],
                    'task_id': record[1],
                    'response_id': record[2],
                    'score': record[3],
                    'comment': record[4],
                    'timestamp': record[5]
                })
            
            return feedback_list
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return []
    
    def analyze_feedback_patterns(self, feedback_list: List[Dict]) -> Dict:
        """Analyze feedback patterns to identify improvement areas."""
        if not feedback_list:
            return {}
        
        # Calculate overall statistics
        total_feedback = len(feedback_list)
        positive_feedback = sum(1 for f in feedback_list if f['score'] > 0)
        negative_feedback = sum(1 for f in feedback_list if f['score'] < 0)
        
        # Calculate component-specific feedback
        summary_feedback = [f for f in feedback_list if f['summary_id']]
        task_feedback = [f for f in feedback_list if f['task_id']]
        response_feedback = [f for f in feedback_list if f['response_id']]
        
        analysis = {
            'total_feedback': total_feedback,
            'positive_ratio': positive_feedback / total_feedback if total_feedback > 0 else 0,
            'negative_ratio': negative_feedback / total_feedback if total_feedback > 0 else 0,
            'summary_feedback_count': len(summary_feedback),
            'task_feedback_count': len(task_feedback),
            'response_feedback_count': len(response_feedback),
            'trend': self._calculate_feedback_trend(feedback_list)
        }
        
        return analysis
    
    def _calculate_feedback_trend(self, feedback_list: List[Dict]) -> str:
        """Calculate the trend of feedback over time."""
        if len(feedback_list) < 10:
            return "insufficient_data"
        
        # Split feedback into recent (last 30%) and older (first 70%)
        split_point = len(feedback_list) // 3
        recent_feedback = feedback_list[:split_point]
        older_feedback = feedback_list[split_point:]
        
        recent_positive = sum(1 for f in recent_feedback if f['score'] > 0)
        older_positive = sum(1 for f in older_feedback if f['score'] > 0)
        
        recent_ratio = recent_positive / len(recent_feedback) if recent_feedback else 0
        older_ratio = older_positive / len(older_feedback) if older_feedback else 0
        
        if recent_ratio > older_ratio + 0.1:
            return "improving"
        elif recent_ratio < older_ratio - 0.1:
            return "declining"
        else:
            return "stable"
    
    def adjust_weights(self, feedback_analysis: Dict):
        """Adjust component weights based on feedback analysis."""
        if not feedback_analysis:
            return
        
        # Simple adjustment logic based on positive feedback ratio
        positive_ratio = feedback_analysis.get('positive_ratio', 0.5)
        
        # If positive feedback is high, increase all weights slightly
        if positive_ratio > 0.7:
            for key in self.weights:
                self.weights[key] = min(1.0, self.weights[key] + self.learning_rate * 0.1)
        # If positive feedback is low, decrease all weights slightly
        elif positive_ratio < 0.3:
            for key in self.weights:
                self.weights[key] = max(0.1, self.weights[key] - self.learning_rate * 0.1)
        
        logger.info(f"Adjusted weights: {self.weights}")
    
    def get_component_recommendations(self) -> Dict:
        """Get recommendations for component improvements."""
        recommendations = {}
        
        # Based on current weights, suggest focus areas
        if self.weights['summarization_quality'] < 0.4:
            recommendations['summarization'] = "Focus on improving summary accuracy and relevance"
        
        if self.weights['task_relevance'] < 0.4:
            recommendations['task_generation'] = "Improve task generation to better match user needs"
        
        if self.weights['response_helpfulness'] < 0.4:
            recommendations['response_generation'] = "Enhance response quality and helpfulness"
        
        return recommendations
    
    def update_pipeline_configuration(self):
        """Update pipeline configuration based on learned weights."""
        try:
            # This would typically communicate with the pipeline API to adjust settings
            # For now, we'll just log the recommended changes
            recommendations = self.get_component_recommendations()
            
            if recommendations:
                logger.info("RL Agent Recommendations:")
                for component, recommendation in recommendations.items():
                    logger.info(f"  {component}: {recommendation}")
            
            # In a real implementation, this would make API calls to adjust the pipeline
            # For example:
            # - Adjust summarization model parameters
            # - Modify task generation thresholds
            # - Tune response generation templates
            
        except Exception as e:
            logger.error(f"Error updating pipeline configuration: {e}")
    
    def process_feedback_loop(self):
        """Main feedback processing loop."""
        logger.info("Starting RL agent feedback processing loop")
        
        # Collect feedback
        feedback_list = self.collect_feedback()
        logger.info(f"Collected {len(feedback_list)} feedback records")
        
        # Analyze feedback patterns
        feedback_analysis = self.analyze_feedback_patterns(feedback_list)
        logger.info(f"Feedback analysis: {feedback_analysis}")
        
        # Adjust weights based on feedback
        self.adjust_weights(feedback_analysis)
        
        # Generate recommendations
        recommendations = self.get_component_recommendations()
        if recommendations:
            logger.info("Recommendations for improvement:")
            for component, recommendation in recommendations.items():
                logger.info(f"  {component}: {recommendation}")
        
        # Update pipeline configuration
        self.update_pipeline_configuration()
        
        # Store feedback history
        self.feedback_history.append({
            'timestamp': datetime.now().isoformat(),
            'feedback_count': len(feedback_list),
            'analysis': feedback_analysis,
            'weights': self.weights.copy()
        })
        
        logger.info("Feedback processing loop completed")
        return {
            'feedback_processed': len(feedback_list),
            'weights': self.weights,
            'recommendations': recommendations,
            'analysis': feedback_analysis
        }
    
    def get_performance_report(self) -> Dict:
        """Generate a performance report based on feedback history."""
        if not self.feedback_history:
            return {"message": "No feedback history available"}
        
        latest_record = self.feedback_history[-1]
        
        return {
            "current_weights": self.weights,
            "latest_analysis": latest_record.get('analysis', {}),
            "total_feedback_cycles": len(self.feedback_history),
            "last_updated": latest_record.get('timestamp', 'Unknown')
        }

# Global RL agent instance
rl_agent = RLAgent()

def start_feedback_processing_loop():
    """Start the feedback processing loop."""
    try:
        result = rl_agent.process_feedback_loop()
        return result
    except Exception as e:
        logger.error(f"Error in feedback processing loop: {e}")
        return {"error": str(e)}

def get_rl_agent_performance():
    """Get RL agent performance report."""
    try:
        return rl_agent.get_performance_report()
    except Exception as e:
        logger.error(f"Error getting RL agent performance: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Run a single feedback processing cycle
    result = start_feedback_processing_loop()
    print(json.dumps(result, indent=2))