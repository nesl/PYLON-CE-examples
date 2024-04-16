class FSM:
    '''Define FSMs to detect complex event patterns'''
    def __init__(self, label):
        self.label = label
        self.init_state = 'init'
        self.final_state = 'final'

    def state_transition(curr_state, curr_input):
        raise NotImplementedError

    def traverse(self, curr_state, input_sequence):
        if curr_state == self.final_state:
            return True
        if len(input_sequence) == 0:
            return False
        curr_input = input_sequence[0]
        next_state = self.state_transition(curr_state, curr_input)
        return self.traverse(next_state, input_sequence[1:])

    def check(self, input_sequence):
        return self.traverse(self.init_state, input_sequence)


class Event0(FSM):
    ''' Event class: Complete dare
        Pattern: Night time user response happens before Morning care (primitive activity 1 happens before 3)'''
    def __init__(self):
        super().__init__(label=0)

    def state_transition(self, curr_state, curr_input):
        if curr_state == 'init':
            if curr_input == 1:
                next_state = 's1'
            else:
                next_state = 'init'
        elif curr_state == 's1':
            if curr_input == 3:
                next_state = 'final'
            else:
                next_state = 's1'
        elif curr_state == 'final':
            if curr_input == 3:
                next_state == 'final'
            else:
                next_state == 'init'
        else:
            raise Exception(f'State "{curr_state}" is not defined.')
        # print(next_state)
        return next_state

class Event1(FSM):
    ''' Event class: Unhygienic process 
        Pattern: Excretion assistance happens before Meal/medication (primitive activity 0 happens before 2)'''
    def __init__(self):
        super().__init__(label=1)

    def state_transition(self, curr_state, curr_input):
        if curr_state == 'init':
            if curr_input == 0:
                next_state = 's1'
            else:
                next_state = 'init'
        elif curr_state == 's1':
            if curr_input == 2:
                next_state = 'final'
            else:
                next_state = 's1'
        elif curr_state == 'final':
            if curr_input == 2:
                next_state == 'final'
            else:
                next_state == 'init'
        else:
            raise Exception(f'State "{curr_state}" is not defined.')
        # print(next_state)
        return next_state
