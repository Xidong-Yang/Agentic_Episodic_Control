from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, envs):
        self.env = envs

    @abstractmethod
    def update_parameters(self):
        raise NotImplementedError()

    def generate_prompt(self, goal, list_actions, deque_obs):
        ldo = len(deque_obs)

        head_prompt = "Possible action of the agent:"
        for action in list_actions:
            head_prompt += " {},".format(action)
        head_prompt = head_prompt[:-1]

        g = " \n Goal of the agent: {}".format(goal)
        obs = ""
        for i in range(ldo):
            obs += " \n Observation : "
            for d_obs in deque_obs[i]:
                obs += "{}, ".format(d_obs)
        return head_prompt + g + obs

    def prompt_modifier(self, prompt: str, dict_changes: dict) -> str:
        """Use a dictionary of equivalence to modify the prompt accordingly."""
        for key, value in dict_changes.items():
            prompt = prompt.replace(key, value)
        return prompt
