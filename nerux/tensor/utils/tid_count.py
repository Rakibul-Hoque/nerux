class Tid_count:
    global_tid_count = 0

    @classmethod
    def get_current(cls):
        return cls.global_tid_count

    @classmethod
    def get_incremented(cls):
        cls.global_tid_count += 1
        return cls.global_tid_count