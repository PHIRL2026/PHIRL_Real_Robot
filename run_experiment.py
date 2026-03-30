import rospy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.BrushEnv import StateSpace, BrushEnv
from demonstrations.DiscreteActionWrapper import DiscreteActions

def main():
    try:
        rospy.init_node('run_testing', anonymous=True)
    except:
        raise Exception("ROS node initialization failed")

    env = BrushEnv(state_space=StateSpace.CARTESIAN)

    observation, info = env.reset()
    print(f"Starting observation: {observation}")

    total_reward = 0

    action_sequence = [DiscreteActions.MOVE_X, DiscreteActions.MOVE_X, DiscreteActions.MOVE_X, 
                       DiscreteActions.MOVE_X_NEG, DiscreteActions.MOVE_X_NEG, DiscreteActions.MOVE_X_NEG, 
                       DiscreteActions.MOVE_Y, DiscreteActions.MOVE_Y, DiscreteActions.MOVE_Y, 
                       DiscreteActions.MOVE_Y_NEG, DiscreteActions.MOVE_Y_NEG, DiscreteActions.MOVE_Y_NEG]
    
    action_sequence = action_sequence * 3

    for step in range(len(action_sequence)):
        action = action_sequence[step]

        # Take the action and see what happens
        print(f"Step: {step}")
        print(f"Taking action: {action}")
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Observation: {observation}")

        total_reward += reward
        
        if truncated or terminated:
            break

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()