from setuptools import setup, find_packages

setup(
    name="f1-pitstop-optimizer",
    version="1.0.0",
    description="Physics-grounded F1 pit stop strategy simulation and RL optimization",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "fastf1>=3.1",
        "gymnasium>=0.29",
        "stable-baselines3>=2.2",
        "torch>=2.0",
        "fastapi>=0.104",
        "uvicorn>=0.24",
        "pydantic>=2.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "jupyter>=1.0", "plotly>=5.17"],
    },
    entry_points={
        "console_scripts": [
            "f1-sim=pitstop.simulation.monte_carlo:__main__",
            "f1-train=pitstop.strategy.rl_agent:__main__",
            "f1-validate=pitstop.data.fastf1_loader:run_full_validation",
            "f1-api=api.main:app",
        ]
    },
)
