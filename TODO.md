# silver system, version 2022.1120 - TODO (draft)

The present document serves as a _post-mortem_ and expands on ideas from the accompanying project report and slides, and augments them with:
- details provided by the Avatar team but not included in the report;
- conclusions drawn from the project's unfolding, implementation and results after the report deadline;
- the authors' research into software engineering good practices, deep learning and computer vision concepts over the summer of 2022, along with the first weeks of the authors' deep learning and explainability (XAI) classes taking place during the Fall semester of 2022/2023.

The purpose of this document is not to bring definitive answers but to try and define avenues of reflection in the eventuality of a continuation of the research project, whether by its authors, their supervisor or any present or future participant.

The term _post-mortem_ merely reflects the fact that this document was drafted after the research project's defense, which took place at the end of May 2022.

**Please note** that this document, in its present draft state, is incomplete and has not yet been reviewed by, and does not in any way reflect the views of or otherwise bind the authors' supervisor for this research project, Associate Professor Denis Sheynikhovich, to whom the authors would like to reiterate their sincere gratitude for his attention and support over the course of their collaboration.

## Implementation details

The following items would represent a first but essential step in extending the project: making sure its codebase is sound and reusable.

### Code linting
Making the code compliant with PEP8 and assorted code legibility standards

### Object-oriented architecture
- encapsulation to improve code legibility, maintainability and reusability
- network- and dataset-agnostic implementation of the method for improved applicability (caveat: based on PyTorch because of the reliance on forward hooks and buffers; transferable to other frameworks such as Keras or Caffe ?)

### PyTorch built-in classes, data structures and other characteristics
PyTorch tensors as optimized structures compared to those used to store activations and sort them to determine the top-k images; more generally limit or downright eliminate reliance on NumPy, to avoid multiplying the import footprint of the code

PyT classes and functions to add a layer of abstraction (see OO architecture above). In particular:
- DataSets and DataLoaders
- a Model class which would be network- (AlexNet or AvatarNet) and dataset-agnostic

Taking advantage of PyTorch's functionalities to make the implementation device-agnostic, i.e. relying on GPU if available, but able to default to CPU execution otherwise, without any user input. If GPU is available, and used by the implementation, we could expect an increase in execution speed.

## Larger context
Questions that the chosen explainability method, AvatarNet's and the Avatar dataset's underlying behaviour modeling approach, and time constraints have not allowed us to address over the course of the research project itself:

### Choice of network
AlexNet was the state of the art at publication time in 2012, and blew its second-place competitor out of the water; however, it has since been outperformed by, among other things, Transformers-based architectures. Would such networks be relevant for the purpose of this research project ? Specifically, how about the biological plausibility of vision Transformers as models for the human visual cortex ?

### Dataset curation and impact on learned representations
Given the redundancies in the dataset, the possible redundancies in the representations learned by the network would be relevant to investigate

### Explainability method
The chosen XAI method works well for geometric patterns but not so much for textures and colors or more generally diffuse stimuli: it allows for localization of the stimulus to which a given unit is succeptible, but not for precise identification or representation. As a consequence, it seems necessary to implement other methods, such as deconvnets, GradCAMs or other methods possibly identified by further investigation into the state of the art.

### Underlying assumptions in the modeling approach
Insofar as the authors, who are not behavioral neuroscientists, understand the underlying Avatar experiment, they would like to raise the following point.

Spatial orientation is, in essence, a sequential task. The authors feel that the conceptual approach behind the training of AvatarNet on static images does not adequately reflect the sequential nature of the task.

This approach assumes that information about the behaviour during the task can be successfully encoded or compressed in a single image taken at a specific point in time during the subject's exploration, in terms of vision and movement, of the experiment's artificial environment, image which is then classified by AvatarNet into one or the other category.

Training a network to classify such static images, and using it as a model for the considered behavioral task, raises the question of what the model actually learns in terms of internal representations, but more importantly of _how relevant these internal representations would be in terms of actual insights_ into the human decisional process during the task.

The authors would like to suggest investigating an alternate approach:
- using a network architecture that can conceptually capture the sequential nature of the stimuli absorbed by a subject during the orientation task;
- subsequently applying a relevant XAI method;
... which combined might give the insights into the human subject's decision process that the Avatar team are looking for.

The exact nature of both technical solutions, to date, remains to be determined (see in particular ``explainability method`` above).

The authors feel that it might be worth exploring, through a future analysis of the state of the art, whether recurrent networks such as computer vision LSTMs or GRUs would be:
- applicable to the Avatar dataset in its extant state (without either data augmentation or additional data acquisition);
- technically feasible and practical (previous implementations ? training time ?);
- biologically plausible as models of the decision process in the human visual cortex.
