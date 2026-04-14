def get_llm_abstruct_prompt(prompt_obs, env_description):
        mission = prompt_obs.split('Goal of the agent: ')[1].split('\n')[0]
        observation = prompt_obs.split('Observation :')[1]

        post_part = f"You are an expert at distilling text-based environment observations into compact, action-oriented state representations.\n\n===== GAME DESCRIPTION =====\n{env_description}\n\n===== MISSION =====\n{mission}\n\n===== INPUT =====\nObservation:\n{observation}\n\n===== INTERNAL REASONING PROCESS =====\nProcess the following steps internally to formulate your response.:\n1. PARSE IMMEDIATE SUB-GOAL: Analyze the mission and observation to determine the immediate sub-goal (e.g., a specific object, location, or entity). Only select a target if its prerequisites are met.\n2. LOCATE TARGET: Scan for the IMMEDIATE SUB-GOAL. Extract its relative position/topological path, stripping away flavor text, exact distances, or numbers.\n3. AGENT STATE: Identify any active states or inventory items the agent currently possesses.\n4. IMMEDIATE ADJACENCY: Identify obstacles, open paths, or entities exactly 1 valid movement step away in primary directions (ignore diagonals).\n5. ACTION READINESS: Determine if the Current Target is within interaction range based on the environment rules (e.g., immediately adjacent or occupying the same space).\n\n===== OUTPUT SPECIFICATION =====\nYou must output ONLY the final state representation using the exact format below. Do not include any conversational filler, explanations, or step-by-step breakdowns. \nIMMEDIATE SUB-GOAL Location: <concise direction/path, or 'unknown'>\nInventory/State: <items held/status, or 'empty'>\nImmediate Adjacency: <Direction: object/open, Direction: object/open...> (or 'unknown')\nInteractable: <yes/no>\n\n===== EXAMPLE =====\n[Example Input]\nGame Description: Go to an object, inside a single room with distractors.\nMission: go to a blue box.\nObservation: You carry a blue key, You see a wall 4 steps forward, You see a blue ball 3 steps left and 1 step forward, You see a blue box 1 step right\n[Example Output]\nIMMEDIATE SUB-GOAL Location: right\nInventory/State: empty\nImmediate Adjacency: Forward: open, Left: open, Right: close\nInteractable: no"
        
        user_content = post_part
        messages = [
                {"role": "system", "content": 'You are a helpful assistant.'},
                {"role": "user", "content": user_content}
            ]
        return messages

def get_llm_abstruct_prompt1(prompt_obs, env_description):
        mission = prompt_obs.split('Goal of the agent: ')[1].split('\n')[0]
        observation = prompt_obs.split('Observation :')[1]

        post_part = f"You are MazeNavigator-GPT, an expert at reading text-based maze observations.\n\n===== GAME DESCRIPTION =====\n{env_description}\n\n===== MISSION =====\n{mission}\n\n===== INPUT =====\nObservation:\n{observation}\n\n===== TASK (do **all** in order, think step‑by‑step)=====\nSTEP 0 - PARSE MISSION\n•Analysis the game description and the observation, determine your **current target object**. If the prerequisites for certain goals are not met, they **should not** be set as current target object. •Extract the exact **color + item** that defines the **current target object**.\nStore it as one pair (e.g. `red ball`). This is the *current target object*.\n\nSTEP 1 — BUILD OUTPUT LIST\n\nA. **Try to find the current target object** in the observation.\n• Scan the single Observation for any mention of that *current target object* **with the same color**.\n• Ignore objects whose does **not** match exactly.\n• For each valid sentence:\n① Remove all numbers (distances, counts, step sizes).\n② Keep the pure **direction phrase** (e.g. `left and forward`, `right`).\n③ Create an entry in the form: `target <direction phrase>`*Example → `target left and forward`*\n• Preserve the order they appear in the Observation.\n\nSTEP 2 — CARRYING ITEM? \n• Output `yes` if the Observation states the agent is carrying anything; else `no`.\n• One word only.\n\n ══════════  STRICT RULES  ══════════\n• Use the literal words **`target`**(lowercase) followed by a space and the direction phrase. \n*Examples* → `target left and forward`\n• Direction phrases must not contain any numbers.\n• Output must be exactly two lines with no trailing spaces.\n\n══════════ OUTPUT FORMAT (exactly 2 lines, no extra spaces) ══════════\n\nLine 1: TARGETS: <comma‑separated list from STEP 1, **may be empty**. Double check that the preconditions for the current target object have been met, if not leave empty.\nLine 2: CARRYING: <yes|no>\n\n."

        user_content = post_part
        messages = [
                {"role": "system", "content": 'You are a helpful assistant.'},
                {"role": "user", "content": user_content}
            ]
        return messages

def get_llm_abstruct_prompt2(prompt_obs, env_description):
        mission = prompt_obs.split('Goal of the agent: ')[1].split('\n')[0]
        observation = prompt_obs.split('Observation :')[1]

        # 当前最优
        post_part = f"You are MazeNavigator-GPT, an expert at reading text-based maze observations.\n\n===== GAME =====\n{env_description}\n\n===== MISSION =====\n{mission}\n\n===== INPUT =====\nObservation:\n{observation}\n\n ===== TASK =====\n\nTask 1 - OBJECT ANALYSIS\nIs there any objects or walls are there in the three directions of the agent:\n'1 step forward without left and right'\n'1 step left without forward'\n'1 step right without forward'\nIf No information provided, output no.\n\nTask 2 - Target 1 step forward?\nOutput `Target 1 step forward: yes` if the **current target object** ONLY at 1 step forward; else if the **current target object** is at 1 step left/right and 1 step forward, output `Target 1 step forward: no`.\n\nDouble check your answer.Rember: Do **NOT consider** the object described by diagonal direction or multi steps, such as `1 step left/right and 1 step forward`. \nAt last, summarize your analysis with a list. Only output the list, no extra text: [At 1 step forward: <yes/no>; At 1 step left: <yes/no>; At 1 step right: <yes/no>; Target 1 step forward: <yes/no>]"
        
        
         
        user_content = post_part
        messages = [
                {"role": "system", "content": 'You are a helpful assistant.'},
                {"role": "user", "content": user_content}
            ]
        return messages

def get_world_model_action_prompt(state, history, world_model, env_description, env_mission):
        history = history.split('Current Observation:')[0]

        prompt = f"You are **MazeNavigator-GPT**, a specialist at solving navigation games.\nYour goal is to choose the single best action based on the mission, the world model, prior history and the current observation.\nAvoid getting stuck in unproductive action loops.\nYou have access to the following actions (tool names are case-sensitive):\n• turn left - rotate 90° left\n• turn right – rotate 90° right\n• go forward - move one step forward  \n• pickup - pick up an item in the same cell\n• drop – drop the carried item\n• toggle – toggle (open / close) an box or door in 1 step forward of you.\n\n Important: You can carry only one item at a time. Before picking up another item, drop the one you are currently carrying in an empty space. \n\n**Game Description:**\n{env_description}\n\n**Mission:**\n{env_mission}\n\n**World Model:**\n{world_model}.\n\n**History:**\n{history}.\n\n**Current Observation:**\n{state}\n\nThink step by step, reason about the information, and choose the best action.\n\nOutput format:\n<reason>Your reason</reason>\n<action>The action you choose</action>."

        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
                ]
        return messages

def get_history_action_prompt(state, history):

        #Do not output your hidden reasoning.Output only the final answer as an integer.
        state = history.split('Current Observation:')[1]
        prompt = f"You are an intelligent agent playing a navigation game.\n\n[Chain of Thought Instructions (for internal reasoning, not to be disclosed externally)]:\n1. Read and understand the 'Game Description' and the 'Mission.'\n2. Combine the 'Historical Observation and Action' and the 'Current observation' to reason step by step internally:\n- Reflect on the environment and how it relates to the Mission.\n- Determine the best next action that will help fulfill the Mission.\n3. In your final answer, output only a single numeric action without revealing these internal reasoning steps.\n\n**Game Description:**\nGo to the red ball. No distractors present.\n\n **Mission:**\ngo to the red ball\n\n**Action Space:**\n0: turn left, 1: turn right, 2: go forward, 3: pickup, 4: drop, 5: toggle;\n\n**Historical Observation and Action:**\n{history}\n\n**Current observation:**\n{state}\n\n[Final Decision Output]:\nPlease provide the single most appropriate numeric action based on the information above, keeping your reasoning hidden."
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
                ]
        return messages

def get_world_model_update_prompt(world_model, previous_observation, action, new_observation):
        # import pdb; pdb.set_trace()
        if world_model:
                if len(world_model)>512:
                        world_model = None

        prompt = f"Objective: The main goal is to meticulously gather information from environment exploration observations and update a structured knowledge graph representing the topology and contents of the world. **Don't include any speculation. Only incorporate new, directly observed information.**\n- **ALWAYS retain all items and triplets from the previous knowledge graph. DO NOT remove or modify any previously recorded entities or relationships, even if they are not observed in the current observation.****\n\n===== INPUT =====\n**Previous Knowledge Graph:**\n{world_model}\n\n**Previous Observation (Part of the observation in the room):\n{previous_observation}\n**Action:**\n{action}\n\n**New Observation (Part of the observation in the room):**\n{new_observation}\n\n===== TASK: Building and Updating the Knowledge Graph =====\nFollow these steps strictly:\n\nStep 1 - Location Tracking and Node Properties (Entities):\n- Assign location labels sequentially as new distinct areas/nodes are confirmed. Name the first location 'Location A', and label subsequent new locations as 'Location B', 'Location C', etc. **If a location has been visited previously, you MUST reuse its existing label.**\n- For the current location, track all **explicitly observed entities/items**.\n- **IMPORTANT MEMORY RULE:** First, import any entities already memorized for that location from the Previous Knowledge Graph. Then, append only the newly observed entities (if not already recorded). **Do not remove any entities from previous observations.**\n- **Ignore exact entity positions or coordinates.**\n- Format: Location A[entity1, entity2, entity3, ...]; ...\n\nStep 2 - Triplet Creation (Connections/Edges):\n- **ONLY perform this step if the Action and New Observation explicitly confirm a physical transition between two distinct locations via a specific path or barrier** (e.g., traversing a door, walking a trail, entering a portal).\n- If no spatial transition occurred (e.g., the agent just picked up an item or looked around), **SKIP THIS STEP entirely and do NOT create any new triplets.**\n- When creating triplets, strictly use the format: `subject, relation, object`.\n- The `subject` is the starting location.\n- The `relation` is the specific connection traversed (include its descriptive state, e.g., 'open wooden door', 'winding dirt path', 'blue portal').\n- The `object` is the newly reached location.\n\n===== OUTPUT SPECIFICATION =====\n- You must return ONLY the updated knowledge graph using the exact format below.\n- **No extra commentary, conversational filler, or explanations are allowed.**\n- The first line must state the current location.\n[Format]\nCurrent Location: Location X\nLocation A[entity1, entity2, ...]; Location B[entity3, ...]; subject_1, relation_1, object_1; subject_2, relation_2, object_2; ..."
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
                ]
        return messages

def get_critical_prompt(prompt, env_description, env_mission):

        prompt = f"You are DecisionAnalyzer-GPT, an expert at evaluating state-action spaces in reinforcement learning and text-based environments.\n\n## Inputs\n**Game Description:**\n{env_description}\n\n**Mission:**\n{env_mission}\n\n**Current Observation:**\n{prompt}\n\n**Task:**\nEvaluate if the current observation represents a **critical state**.\nA state is critical if the choice of action taken right now can substantially change the remaining cumulative return (i.e., the likelihood or efficiency of completing the mission).\n\nTo determine this, consider a state **CRITICAL (yes)** if it involves any of the following:\n1. High-stakes or irreversible choices (e.g., potential death, destroying an item, alerting an enemy).\n2. Significant resource expenditure (e.g., using a consumable item, spending limited currency).\n3. Major topological intersections or branching paths where a wrong choice leads significantly off-track (e.g., a T-junction or multiple locked doors).\n4. Immediate proximity to the final mission target, where the very next action directly determines success.\n\nConsider a state **NON-CRITICAL (no)** if it involves:\n1. Routine movement or forced traversal (e.g., walking down a straight corridor with no other exits).\n2. Safe, generic exploration with no immediate interactable objects of consequence.\n3. States where all available valid actions lead to roughly the same outcome or can be easily reversed without penalty.\n\nBased strictly on these criteria, output 'yes' if it is a critical state, or 'no' if it is not. Do not provide any additional explanations, comments, or text."

        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
                ]
        return messages