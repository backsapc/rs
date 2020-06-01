namespace CCDMF.Models
{
    public class MfMetric
    {
        public MfMetric(double rootMeanSquaredError, double rSquared)
        {
            RootMeanSquaredError = rootMeanSquaredError;
            RSquared = rSquared;
        }

        public double RootMeanSquaredError { get; }
        public double RSquared { get; }
    }
}