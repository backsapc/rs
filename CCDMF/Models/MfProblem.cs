namespace CCDMF.Models
{
    public class MfProblem
    {
        public int M { get; set; }
        public int N { get; set; }
        public long Nnz { get; set; }
        public MfNode[] R { get; set; }
    }
}