/* GC information is stored BEFORE the object structure. */
typedef union _gc_head 
{
    struct {
        union _gc_head *gc_next;
        union _gc_head *gc_prev;
        Py_ssize_t gc_refs;
    } gc;
    long double dummy; /* force worst-case alignment */
} PyGC_Head;

/* this write operation caused memory pages to be Copy-On-Write-ed  */

typedef union _gc_head_ptr
{
    struct {
        union _gc_head *head;
    } gc_ptr;
    double dummy; /* force worst-case alignment */
} PyGC_Head_Ptr;

/* test */

lists = []
strs = []
for i in range(16000):
    lists.append([])
    for j in range(40):
        strs.append(' ' * 8)
        
        
/* test */



# hiding shared objects

static PyObject *
gc_freeze_impl(PyObject *module)
{
    for (int i = 0; i < NUM_GENERATIONS; ++i) {
        gc_list_merge(GEN_HEAD(i), &_PyRuntime.gc.permanent_generation.head);
        _PyRuntime.gc.generations[i].count = 0;
    }
    Py_RETURN_NONE;
}
