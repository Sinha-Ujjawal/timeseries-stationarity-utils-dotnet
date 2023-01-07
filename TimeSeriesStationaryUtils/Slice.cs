using System.Collections;

namespace TimeSeriesStationaryUtils
{
    public class Slice : IEnumerable<uint>
    {
        private uint start;
        private uint step;
        private uint? count;

        public Slice(
            uint start=0,
            uint step=1,
            uint? count=null
        )
        {
            this.start = start;
            this.step = step;
            this.count = count;
        }

        public uint ElementAt(int index)
        {
            return this[(uint) index];
        }

        public uint this[uint index]
        {
            get { return start + (index * step); }
        }

        public IEnumerator<uint> GetEnumerator()
        {
            var values = Extensions.Iterate(s => s + step, start);
            if (count is not null)
                values = values.Take((int) count);
            foreach(var value in values)
                yield return value;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}