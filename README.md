# Coloured trails TOM analysis of LLMs

---

> _The Original `README.md` starts below. This part identifies the additions and changes._

In order to generate a dataset, each player plays 20 games. Half of those are with a "deception" prompt which tells the agent it is alright (and perhaps beneficial) to deceive the human, the other half are not. In addition, for each half, we split the boards into two sets: diagnostic, and normal. The diagnostic boards are generated such that higher order TOM agents are expected to have an advantage.

Games are stored in the `results/panel_experiment/openrouter/` directory. Each player has chosen a seed, with which all randomness is generated, to ensure reproducibility. The games are stored as individual JSONs.

To start a game you need the following command:

```bash
uv run python main.py openrouter --seed ? --advantage-threshold 200
```

`openrouter` tells you that you are playing against the OpenRouter agent. This sends requests to the OpenRouter platform and uses the GPT-OSS 120B model to choose the actions. If you use this agent, you need to set the OpenRouter API key:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

The `--advantage-threshold` tells you the advantage the higher order TOM is expected to have for the diagnostic boards.

Once you start playing a game, you can send an offer by pressing a number key for each chip colour (backspace to undo). These represent the chips you keep. Hitting enter saves the offer. You can then type a message to pass along to the LLM, and then type in some thoughts you have. During the data generation, our thoughts had a set structure.

```
([?|A-E1-5|^[TB][LR][TB][LR]$] )*- [The actual thoughts]
```

The first part encodes where we believe the opponent goal square is. The latter are the actual thoughts.

We can view the results more easily by running the `analyze_panel.py` file.

```bash
python analyze_panel.py openrouter
```

---
Make sure to install all the required dependencies before running.

Usage: python main.py <player\_0> <player\_1>

You can specify a different set of pre-generated boards (or provide your own) by using the --boards-file <file.json> argument.

By default, this setup runs standard ToM and LLM agents alongside passive "shadow agents" that track and analyze every decision. This deep analysis is interesting, but it significantly slows down game execution. To bypass this feature and speed up your runs, comment out lines 218 and 223, and replace them with the empty dictionary initializations (the muted lines right underneath them). Check main.py for additional customizable arguments.

====================================================================================================

\# Colored Trails: A Theory of Mind (ToM) Benchmark for LLM's

This project replicates and documents the Theory of Mind (ToM) trading agents for the Colored Trails game \[1]. The system simulates negotiation between two agents who must exchange resources (colored chips) to traverse a grid board and reach a goal location. The primary focus is on how agents model their opponents' beliefs and goals using recursive reasoning (ToM orders $0, 1, 2$).

\## Game Environment

\### Board \& Chips

The board is a $5$ x $5$ grid of colored tiles. Players receive four randomly selected chips and must try to reach their goal location. The starting position is always at the center of the board, and there are twelve possible goal locations located at the corners, each requiring at least three steps to reach. Moving onto a tile requires spending one chip of that tile’s color, with the exception that the starting tile is free. Players can only move sideways, not diagonally.

\### Scoring

A player's utility $U$ is calculated based on their final position and remaining chips. Players receive $50$ points for reaching their goal, $5$ points per unused chip, and incur a 10-point penalty for each remaining step to the goal.

\### Negotiation

Players enter a negotiation phase before the final scores are determined. The initiator agent proposes a redistribution of chips (Offer $O$). The responder agent can choose one of the following actions:

\- Accept: The trade is executed and the game ends.

\- Reject \& Counter: The trade is rejected, roles swap, and negotiation continues.

\- Withdraw: The negotiation ends with no trade.

Every round of negotiation incurs a cost of $-0.1$

\## Agent Design

\### Zero-Order Agents

The higher-order theory of mind agents are implemented using recursion, and the zero-order agent is the bottom level of this recursion. Zero-order agents do not reason about the mental content of their opponent; instead they form zero-order beliefs about the likelihood of acceptance for potential offers. These zero-order beliefs come in two types: type beliefs and color beliefs.

\##### Type Beliefs

Type beliefs are based on the balance between the amount of chips that are given and received in an offer. The idea is that a zero-order agent can learn that offers in which they receive many chips but offer little in return, are less likely to be accepted.

\##### Color beliefs

Color beliefs are specific probabilities attached to concrete offers (e.g., "Give 1 Black, Receive 2 White"). While type beliefs capture general generosity trends (e.g., "opponent dislikes losing chips"), color beliefs try to capture context-specific goals (e.g., "opponent specifically needs black chips").

\### First-Order Agents

First-order agents apply theory of mind by creating a model of their opponent. This model is essentially a zero-order agent, the only difference being that the opponent's goal location is unknown. Because the first-order agent does not know where the opponent is trying to go on the board, it cannot simply run its internal zero-order model once. Instead, it must treat the opponent's goal as a hidden variable.

\##### Location Beliefs

To handle this uncertainty, the first-order agent maintains a probability distribution over all possible goal locations on the board. If the opponent proposes a trade, the first-order agent adjusts the likelihood of each potential goal location based on that trade.

\### Second-Order Agents

Second-order agents add an extra layer of recursion. They model their opponent as a first-order agent, which itself maintains a zero-order model of the second-order agent. This means the agent assumes that the opponent is also trying to guess its own location. By reasoning through multiple layers of recursion, the second-order agent can see how potential offers would affect the location beliefs of its opponent. Therefore, it might implicitly engage in complex signaling, using false beliefs to manipulate its opponent.

\## Code Implementation Details

\### Belief Implementation

\##### Belief Structure

The type beliefs are implemented as a $5$ x $5$ matrix of singular values between $0$ and $1$, all initialized to $1$. The rows of the matrix correspond to the number of chips received, while the columns correspond to the number of chips offered in return.

The color beliefs are implemented as a list, with one value for each potential offer. Again, the values can range between $0$ and $1$, and they are all initialized to $1$. The length of this list depends on the board and available chips.

Finally, the location beliefs are implemented as a list of probability values representing the likelihood of each tile being the opponent's goal. The list has a length of $12$, corresponding to the number of possible goal locations on the board. Unlike the zero-order beliefs, these values represent a probability distribution and must sum to $1$. They are initialized as a uniform distribution ($1/12$), reflecting that the agent initially considers all locations equally likely.

\##### Belief Persistence

Type beliefs are not specific to individual game states; instead, they capture generalizable patterns. These beliefs are persistent, allowing the agents to learn patterns over the course of hundreds of games. In contrast, color beliefs and location beliefs are strictly game-specific, which is why they are always reset between games. It is important to note that the type beliefs are not directly used in the agents' reasoning process. Instead, they are only used to initialize the color beliefs at the start of each new game. The color beliefs are the ones actually used in the reasoning process.

\##### When are Beliefs Updated

Belief updates are triggered by specific events in the negotiation protocol. The timing differs based on the type of belief.

When the agent receives an offer, it treats this as a positive signal (the opponent likes/would settle for this offer). It increases the belief probability for the offer's type (e.g. three chips offered for one in return) in the type belief matrix, and also raises the offer’s corresponding probability in the color belief list.

When the agent sends an offer, the implementation treats this as a "failed" attempt (rejection) until proven otherwise. It triggers a negative update, decreasing the type and color belief probabilities. If an offer is accepted, the earlier spurious decrease is reversed via an inverse operation, and the type beliefs are updated positively. The color beliefs are not updated after an accepted offer; they will be reset anyway, because the game has ended.

Location beliefs are updated only when the agent receives an offer from the opponent. This is the only moment the opponent reveals information about their hidden goal.

\##### How are Beliefs Updated

Type beliefs are updated using a standard reinforcement learning rule based on a learning speed parameter:

$$

P\_{\\text{new}} = (1 - \\lambda) \\cdot P\_{\\text{old}} + \\lambda \\cdot R \\tag{1}

$$

where $R = 1$ for positive events (receiving/accepting) and $R = 0$ for negative events (sending/rejecting). The learning speed $\\lambda$ ranges between $\[0, 1]$.

The update rules for color beliefs are significantly more complex. Unlike type beliefs, which update a single cell in a matrix, color beliefs update the probabilities of every potential offer in the agent's list simultaneously. For a decrease, the underlying idea is that if one specific offer is rejected, similar or worse offers are also likely to be rejected. In the case of an increase, the logic is similar: if an offer is received (meaning the opponent would accept it), the agent assumes that offers that are strictly worse should have a lower relative probability.

To model the decrease logic, the agent loops through every potential offer in the list and compares it to the rejected offer color by color. For every color where the potential offer allows the agent to keep an equal or greater number of chips than the rejected offer, the belief for that potential offer is penalized using the standard update rule, as shown in (1). Because this check happens per color, an offer that is greedier in multiple dimensions receives cumulative penalties.

Increases are handled in a similar way, but the comparisons are slightly different. For each color, if a potential offer lets the agent keep strictly more chips than the accepted offer, it is penalized. This creates a relative probability increase, where offers that are marginally worse or even better drop slightly, but offers that are significantly worse crash toward zero.

\### Expected Value Calculation

To choose between accepting an offer, making a counter-offer, or withdrawing from negotiations, the agents have compute the expected value ($EV$) of potential offers. The exact calculation depends on the agent's order.

\##### Zero-Order EV

The zero-order agent computes EV directly using its learned color beliefs. For a potential offer $O$:

$$

EV\_0(O) = P \* G + (1 - P) \* -C \\tag 2

$$

where $P$ is the likelihood that the offer $O$ will be accepted by the trading partner (taken directly from the color beliefs list), $G$ is the gain in utility for the agent itself, and $C$ is the cost of a round of negotation ($0.1$). In the code this expression is simplified to:

$$

EV\_0(O) = P(G + C) - C \\tag 3

$$

An important detail is that when the personal utility gain $G$ is not positive, the expected value is immediately set to minus the negotiation cost ($-C$), and therefore does not always fully reflect the true expected value.

\##### First-Order EV

The first-order agent cannot simply look up $P$, because the probability depends on the opponent's unknown location. Instead, it calculates the EV as a weighted average over all possible opponent locations $L$:

$$

EV\_1(O) = \\sum\_{L \\in locations} P(L) \\cdot EV(O | L) \\tag 4

$$

To calculate $EV(O | L)$, the agent performs a one-step lookahead simulation using its internal opponent model. It considers three outcomes:

\- The opponent accepts $O$ right now.

\- The opponent rejects $O$ but returns a counter-offer $O\_{counter}$. The agent checks if receiving $O\_{counter}$ (with an accumulated cost of 2 rounds) is better than withdrawing.

\- The opponent rejects $O$ and ends the game (negotiation cost incurred).

The value is determined by:

$$

EV(O|L) =

\\begin{cases}

G(O) - C \& \\text{if opponent accepts} \\\\

\\max(G(O\_{counter}) - 2 \\cdot C, \\ -C) \& \\text{if opponent counters} \\\\

\-C \& \\text{if opponent withdraws}

\\end{cases} \\tag 5

$$

where $G$ is the potential gain in utility for the agent itself, and $C$ is negotiation cost. Before evaluating each possible opponent location, a copy of the opponent model’s beliefs is created to ensure that any belief updates made during simulation do not affect the agent’s actual beliefs.

\##### Second-Order EV

The second-order agent follows the same summation logic as the first-order agent:

$$

EV\_2(O) = \\sum\_{L \\in locations} P(L) \\cdot EV(O | L) \\tag 6

$$

However, the internal simulation of $EV(O | L)$ is deeper. When the second-order agent simulates the opponent's response, it assumes the opponent is a first-order agent. This means the simulation includes the opponent performing a location belief update on the second-order agent's location before deciding whether to accept or counter. This architecture allows the agent to evaluate offers based on their signaling value (e.g., "If I make this offer, the opponent will think I am at location X, which makes them more likely to accept").

\##### Myopia

It is important to note that while the agents exhibit extensive recursive reasoning, the possess a shallow strategic horizon (modeling the game's future). In the current implementation, agents only evaluate the immediate utility of the opponent's direct counter-offer. On the contrary, human players possess a deep strategic horizon, and would realize that they can haggle. Instead of treating the immediate counter-offer as an ultimatum, they would recognize that rejection paired with a counter-offer is often just the beginning of a multi-stage process to find a mutually beneficial middle ground.

<!-- ##### Confidence

ALSO WRITE ABOUT SELF\_MODEL HERE. The first-order and second-order agents do not rely exclusively on their recursive models. Instead, they maintain a confidence parameter ($0.0$ to $1.0$). This value dictates how much weight the agent places on its sophisticated ToM model versus its simpler internal zero-order model. Confidence = 0: The agent ignores its ToM simulation and behaves exactly like a zero-order agent. Confidence = 1: The agent relies entirely on its recursive simulations.Update Rule: At the end of every interaction, the agent compares the opponent's actual move to what its ToM model predicted. If the model predicted correctly, confidence increases; if it failed, confidence decreases. -->

\## References

\[1] de Weerd, H., Verbrugge, R., \& Verheij, B. (2017). Negotiating with other minds: the role of recursive theory of mind in negotiation with incomplete information. \*Autonomous Agents and Multi-Agent Syst\*, 31, 250-287. <https://doi.org/10.1007/s10458-015-9317-1>
