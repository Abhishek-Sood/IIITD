# Coreset Algorithms Implementation

This repository contains Python implementations of algorithms related to coresets, as per Problem Statement 3 for the IIIT Delhi EPOCH Hackathon. These implementations focus on streamlining data processing and enhancing computational efficiency. The algorithms are based on the research paper "Practical Coreset Constructions for Machine Learning" by Olivier Bachem, Mario Lucic, and Andreas Krause from the Department of Computer Science, ETH Zurich.

## Content
1. [Problem Statement](#problem-statement)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Screenshots](#screenshots)
6. [Demo](#demo)
7. [Contributors](#contributors)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

## Problem Statement
The problem statement required implementing the algorithms related to coresets and understanding the definition of a 'coreset' along with the operations in algo-1 and algo-2 from the mentioned research paper. The task also included comparing the relative error performance of practical and uniform sampling-based coresets.

## Repository Structure

| File Name                 | Description                                                                                                              |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `Uniform_Sampling.py`     | Implements a fixed size coreset construction using uniform sampling and evaluates the relative error performance.         |
| `wkpp.py`                 | Implements weighted kMeans++ necessary for reporting the performance of coresets.                                         |
| `skeleton.py`            | Provides a basic structure for writing the code and can be used as a reference to implement the algorithms.              |
| `image_segmentation.py`   | Demonstrates image segmentation on the coreset compared to image segmentation on the original file.                      |

## Installation

1. Clone the repository:

   ```bash 
   git clone https://github.com/Abhishek-Sood/IIITD.git
   ```
   

2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```
   

## Usage

1. Run the main script to evaluate the relative error performance:

   ```bash
   python Uniform_Sampling.py
   ```
   

2. Run the image segmentation script to visualize the difference between segmentation on the coreset and the original file:

   ```bash
   python image_segmentation.py
   ```
   

## Screenshots
Screenshots demonstrating the implementation.
![Alt text](/screen.png?raw=true "Screenshot")

## Demo

Demo

## Contributors

*Team OtakuGang* :
  - Abhishek Sood
  - Ashish Bachan
  - Sahil Kumar
  - Varun Sharma

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Abhishek-Sood/IIITD/blob/main/LICENSE) file for details.

## Acknowledgments

We would like to thank IIIT Delhi for organising the EPOCH hackathon.   
Furthermore, we wish to express our sincere appreciation to CyFuse, IIIT Delhi for hosting and managing the event.

For more details, refer to the research paper "Practical Coreset Constructions for Machine Learning" by Olivier Bachem, Mario Lucic, and Andreas Krause.
