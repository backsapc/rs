using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CCDMF;
using CCDMF.Logic;
using CCDMF.Models;
using CommandLine;

namespace MfConsole
{
    [Verb("train", HelpText = "Train model with specified input.")]
    class TrainOptions
    {
        //normal options here
        [Option('i', "input", Required = true, HelpText = "Set input file path.")]
        public string Input { get; set; }

        //normal options here
        [Option('o', "output", Required = true, HelpText = "Set output folder path.")]
        public string Output { get; set; }

        //normal options here
        [Option('t', "test", Required = false, HelpText = "Set test input file path.")]
        public string Test { get; set; }

        //normal options here
        [Option('n', "thread", Required = false, HelpText = "Set amount of the threads.")]
        public int Threads { get; set; }

        //normal options here
        [Option('r', "rank", Required = false, HelpText = "Set approximation rank.")]
        public int Factors { get; set; }

        //normal options here
        [Option('v', "verbose", Required = false, HelpText = "Set output to verbose messages.")]
        public bool Verbose { get; set; }

        //normal options here
        [Option('e', "eps", Required = false, HelpText = "Set precision parameter value.")]
        public float Eps { get; set; }

        //normal options here
        [Option('l', "lambda", Required = false, HelpText = "Set regularization parameter value.")]
        public float Lambda { get; set; }

        //normal options here
        [Option('p', "positive", Required = false, HelpText = "Do non negative matrix factorization.")]
        public bool NonNegativeMatrixFactorization { get; set; }

        //normal options here
        [Option('c', "iterations", Required = false, HelpText = "Set number of iterations.")]
        public int NumberOfIterations { get; set; }
    }

    [Verb("predict", HelpText = "Predict ratings with provided model and training data set.")]
    class PredictOptions
    {
        //normal options here
        [Option('i', "input", Required = true, HelpText = "Set input file path.")]
        public string Input { get; set; }

        //normal options here
        [Option('o', "output", Required = true, HelpText = "Set output folder path.")]
        public string Output { get; set; }

        //normal options here
        [Option('m', "model", Required = true, HelpText = "Set model file path.")]
        public string Model { get; set; }
    }

    [Verb("prepare", HelpText = "Prepare provided data to program format.")]
    class PrepareOptions
    {
        //normal options here
        [Option('i', "input", Required = true, HelpText = "Set input file path.")]
        public string Input { get; set; }

        //normal options here
        [Option('o', "output", Required = true, HelpText = "Set output folder path.")]
        public string Output { get; set; }

        //normal options here
        [Option('d', "delimiter", Default = " ", Required = false, HelpText = "Set delimiter symbol.")]
        public string Delimiter { get; set; }
    }


    class Program
    {
        static int Main(string[] args)
        {
            //MainMain();
            //return 0;
            return CommandLine.Parser.Default.ParseArguments<TrainOptions, PredictOptions, PrepareOptions>(args)
                .MapResult(
                    (TrainOptions opts) => RunTrain(opts),
                    (PredictOptions opts) => RunPredict(opts),
                    (PrepareOptions opts) => RunPrepare(opts),
                    errs =>
                    {
                        Console.ReadKey();
                        return 0;
                    });
        }

        private static int RunPrepare(PrepareOptions opts)
        {
            string inputFile = opts.Input;
            string outputTrain = Path.Combine(opts.Output, "train.ml.ratings");
            string outputTest = Path.Combine(opts.Output, "test.ml.ratings");
            string delimiter = opts.Delimiter;

            var ratings = File.ReadAllLines(inputFile);

            var rnd = new Random();

            IList<string> GetRatings(int count) => ratings.OrderByDescending(x => rnd.Next()).Take(count).ToList();

            var training = GetRatings((int) ((ratings.Length) * 0.8));
            var test = GetRatings((int) ((ratings.Length) * 0.2));

            IList<string> SortByUserId(IList<string> rates) =>
                rates.Select(x => x.Split(delimiter))
                    .OrderBy(x => int.Parse(x[0]))
                    .GroupBy(x => x[0])
                    .SelectMany(x => x.OrderBy(y => int.Parse(y[1])))
                    .Select(x => string.Join(" ", x[0], x[1], x[2])).ToList();

            var orderedTraining = SortByUserId(training);
            var orderedTest = SortByUserId(test);

            File.WriteAllLines(outputTrain, orderedTraining);
            File.WriteAllLines(outputTest, orderedTest);

            return 0;
        }

        private static int RunPredict(PredictOptions opts)
        {
            string resultFileName = Path.Combine(opts.Output, "ml.result");
            string inputFileName = opts.Input;
            string modelFileName = opts.Model;

            var problem = IOHelper.LoadDataFromTextFile(inputFileName);
            var model = IOHelper.LoadModelFromFile(modelFileName);

            var predictor = new MfPredictor(model);

            using (var predictionWriter = File.CreateText(resultFileName))
            {
                foreach (var mfNode in problem.R)
                {
                    predictionWriter.WriteLine(Math.Round(predictor.Predict(mfNode.U, mfNode.V), 1));
                }
            }

            return 0;
        }

        private static int RunTrain(TrainOptions opts)
        {
            var trainingPath = opts.Input;
            var testingPath = opts.Test;
            string resultFileName = Path.Combine(opts.Output, "ml.result");

            MfProblem training = IOHelper.LoadDataFromTextFile(trainingPath);
            MfProblem testing = !string.IsNullOrEmpty(testingPath) ? IOHelper.LoadDataFromTextFile(testingPath) : null;

            MfModel model = new MfTrainer().Fit(training, testing, new MfTrainerOptions()
            {
                NumberOfThreads = opts.Threads,
                ApproximationRank = opts.Factors,
                Eps = opts.Eps,
                LambdaRegularization = opts.Lambda,
                NonNegativeMatrixFactorization = opts.NonNegativeMatrixFactorization,
                NumberOfIterations = opts.NumberOfIterations,
                Verbose = opts.Verbose
            });

            var predictor = new MfPredictor(model);

            model.SaveModelToFile(resultFileName);

            if (testing != null)
            {
                MfMetric metrics = predictor.Evaluate(testing);
                Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
            }

            return 0;
        }

        private static void MainMain()
        {
            var data = "Data";
            var trainingPath = Path.Combine(data, "training.ratings");
            var testingPath = Path.Combine(data, "test.ratings");

            MfProblem training = IOHelper.LoadDataFromTextFile(trainingPath);
            MfProblem testing = IOHelper.LoadDataFromTextFile(testingPath);

            Console.WriteLine("Model training started.");

            var model = new MfTrainer(new ConsoleLogger()).Fit(training, testing, new MfTrainerOptions()
            {
                Verbose = true,
                LambdaRegularization = 0.2f,
                NumberOfThreads = 16,
                NumberOfIterations = 8
            });

            var predictor = new MfPredictor(model);

            Console.WriteLine("Prediction calculation started.");

            MfMetric metrics = predictor.Evaluate(testing);

            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
            Console.WriteLine($"RSquared: {metrics.RSquared}");
            Console.WriteLine("Press any key to close..");
            Console.ReadKey();
        }
    }

    class ConsoleLogger : ILogger
    {
        public void Write(string s, params object[] d)
        {
            Console.WriteLine(s, d);
        }

        public void Write(string s)
        {
            Console.WriteLine(s);
        }

        public void WriteLine(string s, params object[] d)
        {
            Console.WriteLine(s, d);
        }

        public void WriteLine(string s)
        {
            Console.WriteLine(s);
        }
    }
}