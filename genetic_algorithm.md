# Working with large virtual chemical libraries: Part 2 - Genetic Algorithms

This is part 2 of a a planned three post series on working with large chemical libraries.
The notebook used to create this post and all the files can be found in [this github repo](https://github.com/jonswain/genetic-algorithms-for-searching-ul-libraries).


The first post in this series on active learning can be [found here](https://jonswain.github.io/active%20learning/ai/cheminformatics/data%20science/machine%20learning/ultra-large%20libraries/2024/05/18/ultra-large-libraries-part-1.html).

---

## Combinatorial libraries

Combinatorial libraries can grow quickly, combining three sets of 1,000 building blocks allows you to access one billion possibilities. For example, 1,000 amines, 1,000 halide functionalised carboxylic acids, and 1,000 boronic acids can be combined to make a billion-member library. That would be a lot of synthesis and a lot of testing if you wanted to screen every member of the library! Even if a computational method took 1 second to score each virtual compound, it would take 32 years to score the entire library. Make-on-demand chemical suppliers such as Enamine have libraries that contain multiple billions of compounds, often created by combining sets of building blocks using reliable chemistry. We need methods for effectively sampling this chemical space.

In this post I'll be looking into using a genetic algorithm to search a small combinatorial library. These are quick and incredibly simple methods that can be used to search combinatorial libraries for the combination of building blocks that maximises or minimises a scoring function.

Whilst this method uses a combinatorial library that contains all possible configurations of all three building blocks, it's possible to extend it to more complex combinatorial libraries by [storing the building blocks and reactions as a graph](https://www.youtube.com/watch?v=lNzW6_z_jko). 

## Genetic algorithm

Genetic algorithms are biologically inspiried, based on biological natural selection. They use a very simple algorithm, with no machine learning or complex statistics. One of the big advantages is that you don't need to fully enumerate of score the entire library, only certain combinations of building blocks need to be enumerated and scores, reducing memory and computational costs. Working in building block space rather than with enumerated structures means the complexity will scale with number of building blocks, which will increase much more slowly than the total library size. The downside of the genetic algorithm is that it requires combinatorial libraries, and won't work with large collections of diverse molecules not made up from the same building blocks.

Initially a random population of the unlabelled data is selected by randomly choosing building blocks, and enumerating the reaction products from these building blocks.

The genetic algorithm has a cycle that is repeated and each round is called a generation.
1. The population is labelled using the expensive scoring function.
2. A selection pressure is applied to the population (the lowest scoring in the population are removed).
3. A new population is created by randomly shuffling the building blocks from the surviving population (mating).
4. Mutations (random building blocks) are added to the population to prevent getting stuck in a local minimum.
5. The above steps are repeated on the new population until a finish criterion is met.

---

## Imports

First we need to import the libraries we will be using.


```python
import math
import time
from functools import partial
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from tqdm import tqdm
```

## Expensive scoring function

The first function we will define is the expensive scoring function, this will take a list of RDKit molecules and return a list of scores. As in the previous example, I'm going to try find the compound from within the library with the lowest calculated Log P (cLogP). This is actually a very fast calculation and can be done exhaustively, which means we can confirm if the genetic algorithm is finding the lowest value and triggering the early stopping.


```python
def calc_logp(mols: list[Chem.Mol]) -> list[float]:
    """Calculate the logP value for a list of compounds.

    Args:
        mols (list[Chem.Mol]): The molecules.

    Returns:
        list[float]: The scores of the molecules.
    """
    return [Descriptors.MolLogP(mol) for mol in mols]
```

## Defining some useful functions

Next, we need to define some useful functions for the active learning pipeline.

The first function takes the indices of three building blocks and enumerates the product of their reaction. Since not all combinations are possible, it will return None if the reaction fails. The second function takes the surviving population from the selection pressure and creates a new population by shuffling the building blocks. The third takes the new shuffled population and adds in random mutations, this prevents the algorithm getting stuck in a local minimum by ensuring building block diversity.


```python
def make_molecule(
    r1: int,
    r2: int,
    r3: int,
    building_blocks: list[list[str]],
    rxn: Chem.rdChemReactions.ChemicalReaction,
) -> Chem.Mol | None:
    """React building blocks to make a molecule.

    Args:
        r1 (int): Index of the first building block.
        r2 (int): Index of the second building block.
        r3 (int): Index of the third building block.
        building_blocks (list[list[str]]): The building blocks.
        rxn (Chem.rdChemReactions.ChemicalReaction): The reaction to combine the
                                                     building blocks.


    Returns:
        Chem.Mol | None: The molecule or None if the reaction fails.
    """
    bbs = [building_blocks[0][r1], building_blocks[1][r2], building_blocks[2][r3]]
    reagent_mol_list = [Chem.MolFromSmiles(x) for x in bbs]
    products = rxn.RunReactants(reagent_mol_list)
    if products:
        Chem.SanitizeMol(products[0][0])
        return products[0][0]
    return None


def shuffle_population(data: pd.DataFrame, population_size: int) -> pd.DataFrame:
    """Shuffle the population building blocks.

    Args:
        data (pd.DataFrame): The surviving population data.
        population_size (int): The size of the population.

    Returns:
        pd.DataFrame: The shuffled population.
    """
    r1s = data["r1"].to_list()
    r2s = data["r2"].to_list()
    r3s = data["r3"].to_list()
    new_population = pd.DataFrame(
        {
            "r1": [np.random.choice(r1s) for _ in range(population_size)],
            "r2": [np.random.choice(r2s) for _ in range(population_size)],
            "r3": [np.random.choice(r3s) for _ in range(population_size)],
        }
    )
    return new_population


def mutate_population(
    data: pd.DataFrame, mutation_rate: float, building_blocks: list[list[str]]
) -> pd.DataFrame:
    """Mutate the population building blocks.

    Args:
        data (pd.DataFrame): The surviving population data.
        mutation_rate (float): The fraction of building blocks to mutate.
        building_blocks (list[list[str]]): The building blocks.

    Returns:
        pd.DataFrame: The mutated population.
    """
    for count, column in enumerate(data.columns):
        selection = data.sample(frac=mutation_rate)
        data.loc[selection.index, column] = np.random.randint(
            0, len(building_blocks[count]), len(selection)
        )
    return data
```

## Genetic algorithm pipeline

The genetic algorithm needs the lists of building blocks, a reaction to couple them to enumerate products, and a function to score the molecules. It also has a number of hyperparameters than can be tuned:

- `population_size`: The number of molecules in each generation to try enumerate. A larger population will have a greater diversity of building blocks, but will take longer to enumerate and score.
- `num_generations`: The number of cycles of the genetic algorithm to run. More generations will give a better chance of finding the optimum value, but will take longer to run.
- `selection_pressure`: The fraction of the population to remove each cycle. A stronger selection pressure will speed up selection, but increase the chance of being caught in a local minimum.
- `mutation_rate`: The proportion of building blocks in the new population to replace with a random building block. A too high value will stop the algorithm finding the best combination, but a too low value will also increase the chance of being caught in a local minimum.


```python
def run_genetic_algorithm(
    building_blocks: list[list[str]],
    rnx: Chem.rdChemReactions.ChemicalReaction,
    population_size: int,
    num_generations: int,
    selection_pressure: float,
    mutation_rate: float,
    scoring_function: Callable,
    minimize: bool,
    early_stopping_value: float | None = None,
) -> pd.DataFrame:
    """Run the genetic algorithm.

    Args:
        building_blocks (list[list[str]]): The building blocks.
        rnx (Chem.rdChemReactions.ChemicalReaction): The reaction to combine the
                                                     building blocks.
        population_size (int): The size of the population in each generation.
        num_generations (int): The number of generations to run.
        selection_pressure (float): The fraction of the population to discard.
        mutation_rate (float): The fraction of the population to mutate.
        scoring_function (Callable): The function to score the molecules.
        minimize (bool): Whether to minimize the scoring function.
        early_stopping_value (float | None, optional): The value to stop early at.
                                                       Defaults to None.

    Returns:
        pd.DataFrame: The history of the population.
    """
    # To keep track of the population each generation
    history = pd.DataFrame()

    # Choose initial population
    population = pd.DataFrame(
        {
            "r1": np.random.randint(0, len(building_blocks[0]), population_size),
            "r2": np.random.randint(0, len(building_blocks[1]), population_size),
            "r3": np.random.randint(0, len(building_blocks[2]), population_size),
        }
    )

    for generation in range(num_generations):
        # Generate molecules
        population["ROMol"] = population.apply(
            lambda x: make_molecule(x["r1"], x["r2"], x["r3"], building_blocks, rnx),
            axis=1,
        )

        # Kill off fatal mutants
        population = population.dropna().reset_index(drop=True)

        # Score molecules
        population["score"] = scoring_function(population["ROMol"].to_list())

        # Save population for analysis
        population["generation"] = generation
        history = pd.concat([history, population])

        # Early stopping
        if early_stopping_value:
            if minimize:
                if round(population["score"].min(), 5) <= round(
                    early_stopping_value, 5
                ):
                    break
            else:
                if round(population["score"].max(), 5) >= round(
                    early_stopping_value, 5
                ):
                    break

        # Select top performing molecules
        population = (
            population.sort_values("score", ascending=minimize)
            .head(int(population_size * (1 - selection_pressure)))
            .reset_index(drop=True)
        )

        # Shuffle and mutate
        population = shuffle_population(population, population_size)
        population = mutate_population(population, mutation_rate, building_blocks)
        population = population.drop_duplicates().reset_index(drop=True)

    return history
```

This function fully enumerates a virtual library by combining three sets of building blocks. The smi files used here were borrowed from [Pat Walters repository on Thompson sampling](https://github.com/PatWalters/TS). This is not necessary for the genetic algorithm, but will allow us to see if the genetic algorithm if finding the best values.


```python
def build_virtual_library() -> pd.DataFrame:
    """Build a virtual library by coupling building blocks from the input smi files.

    Returns:
        pd.DataFrame: The virtual library.
    """
    try:
        library = pd.read_csv("data/library.csv", index_col="smiles")
        library["mol"] = [Chem.MolFromSmiles(s) for s in tqdm(library.index.to_list())]
    except FileNotFoundError:
        reaction_smarts = "N[c:4][c:3]C(O)=O.[#6:1][NH2].[#6:2]C(=O)[OH]>>[C:2]c1n[c:4][c:3]c(=O)n1[C:1]"
        bb_types = ["aminobenzoic", "carboxylic_acids", "primary_amines"]
        rxn = AllChem.ReactionFromSmarts(reaction_smarts)

        building_blocks = []
        for bb in bb_types:
            smil = []
            with open(Path(f"data/{bb}_100.smi"), "r") as f:
                for line in f.readlines():
                    smiles, _ = line.split()
                    smil.append(smiles)
            building_blocks.append(smil)

        total_prods = math.prod([len(x) for x in building_blocks])

        product_list = []
        for reagents in tqdm(product(*building_blocks), total=total_prods):
            reagent_mol_list = [Chem.MolFromSmiles(x) for x in reagents]
            products = rxn.RunReactants(reagent_mol_list)
            if products:
                Chem.SanitizeMol(products[0][0])
                product_list.append(products[0][0])

        library = pd.DataFrame(
            product_list,
            index=[Chem.MolToSmiles(m) for m in product_list],
            columns=["mol"],
        )
        library.to_csv("data/library.csv")
    return library


# Chemistry parameters
RXN = AllChem.ReactionFromSmarts(
    "N[c:4][c:3]C(O)=O.[#6:1][NH2].[#6:2]C(=O)[OH]>>[C:2]c1n[c:4][c:3]c(=O)n1[C:1]"
)
BUILDING_BLOCKS = []
for bb in ["aminobenzoic", "carboxylic_acids", "primary_amines"]:
    smil = []
    with open(Path(f"data/{bb}_100.smi"), "r") as f:
        for line in f.readlines():
            smiles, _ = line.split()
            smil.append(smiles)
    BUILDING_BLOCKS.append(smil)
```

---

## Example 1: Finding the compound with the lowest cLogP

The genetic algorithm was used to find the lowest cLogP from within the library. This was repeated 10 times with no tuning of the hyperparameters which may improvement performance. The genetic algorithm reliably finds the combination of building blocks with the lowest cLogP in the combinatorial library, nearly 100 times faster than enumerating the entire library. On average the genetic algorithm only had to enumerate and score 600 combinations before it found the best scoring combination. 


```python
# GA parameters
POPULATION_SIZE = 500
NUM_GENERATIONS = 20
SELECTION_PRESSURE = 0.5
MUTATION_RATE = 0.1
MINIMIZE = True
NUMBER_OF_REPEATS = 10

# Create the virtual library
virtual_library_start = time.time()
print("Creating virtual library")
library = build_virtual_library()
virtual_library_end = time.time()
print(
    f"Virtual library created in {virtual_library_end - virtual_library_start:.2f} seconds",
)

# Find the minimum logP in the library
logp_start = time.time()
print("Calculating logP values for the library")
all_clogp_values = [Descriptors.MolLogP(mol) for mol in tqdm(library.mol.to_list())]
logp_end = time.time()
print(f"Minimum logP in the library: {min(all_clogp_values):.2f}")
print(f"LogP calculations took {logp_end - logp_start:.2f} seconds")

# Run the genetic algorithm
print(f"Running genetic algorithm {NUMBER_OF_REPEATS} times")
ga_times = []
max_generations = []
for _ in tqdm(range(NUMBER_OF_REPEATS)):
    ga_start = time.time()
    logp_ga_run = run_genetic_algorithm(
        building_blocks=BUILDING_BLOCKS,
        rnx=RXN,
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        selection_pressure=SELECTION_PRESSURE,
        mutation_rate=MUTATION_RATE,
        scoring_function=calc_logp,
        minimize=MINIMIZE,
        early_stopping_value=min(all_clogp_values),
    )
    ga_end = time.time()
    ga_times.append(ga_end - ga_start)
    max_generations.append(logp_ga_run.generation.max())
print(
    f"Genetic algorithm run took {np.average(ga_times):.2f} seconds on average, with a maximum of {np.max(max_generations)} generations"
)

# Calculate improvement
ga_time = ga_end - ga_start
virtual_library_time = virtual_library_end - virtual_library_start
logp_time = logp_end - logp_start
improvement = (virtual_library_time + logp_time) / ga_time
print(
    f"GA was {improvement:.2f} times faster than building and scoring the virtual library"
)
```

    Creating virtual library


    100%|██████████| 1000000/1000000 [01:48<00:00, 9183.87it/s]


    Virtual library created in 120.60 seconds
    Calculating logP values for the library


    100%|██████████| 132500/132500 [00:27<00:00, 4803.96it/s]


    Minimum logP in the library: -5.00
    LogP calculations took 27.59 seconds
    Running genetic algorithm 10 times


    100%|██████████| 10/10 [00:14<00:00,  1.41s/it]

    Genetic algorithm run took 1.41 seconds on average, with a maximum of 12 generations
    GA was 81.90 times faster than building and scoring the virtual library


    


---

## Example 2: Recovering a random compound from within the library using Tanimoto similarity

A reference molecule was randomly selected from the library, and the genetic algorithm was used to recover it from the library by maximising the Tanimoto similarity. In the active learning experiments, this was failing, possibly due to the low number of compounds from within the library with high Tanimoto similarities. The genetic algorithm was able to recover the reference molecule every time, taking less than a second to do so each time.


```python
def calc_similarity(comparison_mols: list[Chem.Mol], ref_mol: Chem.Mol) -> list[float]:
    """Calculate the Tanimoto similarity to a reference compound.

    Args:
        comparison_mols (list[Chem.Mol]): List of molecules to compare.
        ref_mol (Chem.Mol): The reference molecule.

    Returns:
        list[float]: List of similarity scores
    """
    fpgen = AllChem.GetMorganGenerator()
    ref_fp = fpgen.GetFingerprint(ref_mol)
    comparison_fps = [fpgen.GetFingerprint(x) for x in comparison_mols]
    return [DataStructs.FingerprintSimilarity(ref_fp, x) for x in comparison_fps]


# Score the virtual library
print("Scoring the virtual library")
tanimoto_start = time.time()
all_tanimoto_values = calc_similarity(
    library.mol.to_list(), library.mol.sample().values[0]
)
tanimoto_end = time.time()
print(f"Tanimoto calculations took {tanimoto_end - tanimoto_start:.2f} seconds")

# GA parameters
POPULATION_SIZE = 500
NUM_GENERATIONS = 10
SELECTION_PRESSURE = 0.5
MUTATION_RATE = 0.1
MINIMIZE = False
NUMBER_OF_REPEATS = 10

# Run the genetic algorithm
print(f"Running genetic algorithm {NUMBER_OF_REPEATS} times")
ga_times = []
max_generations = []
for _ in tqdm(range(NUMBER_OF_REPEATS)):
    # Choose random reference molecule
    ref_mol = library.mol.sample().values[0]

    # Run the genetic algorithm
    ga_start = time.time()
    tanimoto_ga_run = run_genetic_algorithm(
        building_blocks=BUILDING_BLOCKS,
        rnx=RXN,
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        selection_pressure=SELECTION_PRESSURE,
        mutation_rate=MUTATION_RATE,
        scoring_function=partial(calc_similarity, ref_mol=ref_mol),
        minimize=MINIMIZE,
        early_stopping_value=1,
    )
    ga_end = time.time()
    ga_times.append(ga_end - ga_start)
    max_generations.append(tanimoto_ga_run.generation.max())
print(
    f"Genetic algorithm run took {np.average(ga_times):.2f} seconds on average, with a maximum of {np.max(max_generations)} generations"
)

# Calculate improvement
tanimoto_time = tanimoto_end - tanimoto_start
improvement = tanimoto_time / np.average(ga_times)
print(
    f"GA was {improvement:.2f} times faster than scoring the virtual library with Tanimoto similarity"
)
```

    Scoring the virtual library
    Tanimoto calculations took 7.41 seconds
    Running genetic algorithm 10 times


    100%|██████████| 10/10 [00:07<00:00,  1.29it/s]

    Genetic algorithm run took 0.77 seconds on average, with a maximum of 9 generations
    GA was 9.60 times faster than scoring the virtual library with Tanimoto similarity


    



```python

```
