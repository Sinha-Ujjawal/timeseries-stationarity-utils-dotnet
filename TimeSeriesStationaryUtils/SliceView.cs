namespace TimeSeriesStationaryUtils
{
    public class SliceView<Collection, Index, ValueType>
    {
        private readonly Collection collection;
        private readonly Func<Collection, Index, ValueType> getter;
        private readonly Action<Collection, Index, ValueType> setter;

        public SliceView(
            Collection collection,
            Func<Collection, Index, ValueType> getter,
            Action<Collection, Index, ValueType> setter
        )
        {
            this.collection = collection;
            this.getter = getter;
            this.setter = setter;
        }

        public ValueType this[Index index]
        {
            get { return getter(collection, index); }
            set { setter(collection, index, value); }
        }
    }
}