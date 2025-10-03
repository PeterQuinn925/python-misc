model = 'gpt-oss:20b'
#model = 'gemma3:27b'
#model = 'qwen3:14b'
temperature = 1.2


import subprocess
import threading
import queue
import time

def read_output(pipe, output_queue):
    """
    Reads from the subprocess's output pipe in a separate thread.
    """
    for line in iter(pipe.readline, b''):
        output_queue.put(line)
    pipe.close()

def play_zork():
    """
    Starts the Zork interpreter and interacts with it.
    """
    # Start the frotz process with the game file
#         ['frotz', 'zork_285.z5'],
    proc = subprocess.Popen(
        ['snap', 'run','zork'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text = True,
        bufsize = 1
    )

    # Create a queue and a thread to read the process's output asynchronously
    output_queue = queue.Queue()
    reader_thread = threading.Thread(target=read_output, args=(proc.stdout, output_queue))
    reader_thread.daemon = True  # Thread will close when the main program exits
    reader_thread.start()
    def get_game_output():
        """
        Retrieves all available output from the game.
        """
        output = ''
        time.sleep(0.1)  # Give the process a moment to write output
        while not output_queue.empty():
            output += output_queue.get()
        return output

    def send_command(command):
        """
        Sends a command to the game's input.
        """
        proc.stdin.write(command + '\n')
        proc.stdin.flush()

    # --- Main game interaction loop ---

    # Example sequence of commands
    #commands = ['open mailbox', 'take leaflet', 'open leaflet', 'n', 'w']
    last_response = ""
    while True:
        game_response = get_game_output().strip()
        if len(game_response)>2:
            print(game_response, end='')
            prompt = "Here is what the game is telling you:"+game_response+" What command should we give the game? Return only the command"
            #send game response to LLM
            cmd = query_ollama(model,prompt).message.content
            print("\n",cmd,"\n")
            send_command(cmd)
            if last_response == game_response: #we haven't moved, let's increase the temperature
                temperature = temperature + 0.1
            last_response = game_response

    # Clean up
    proc.terminate()
    reader_thread.join(timeout=1)
import ollama
def setup_ollama(model,prompt):
    response = ollama.chat(
    model=model,
    messages=[{'role': 'system', 'content': prompt}],
    options={
         'temperature': 1.0,  # Range: 0.0 to 1.0
    })
    return response

def query_ollama(model, prompt):
    #print ("prompt******",prompt,"*****")
    response = ollama.chat(
       model=model,
       messages=[{'role': 'user', 'content': prompt}],
       options={
            'temperature': temperature,  # Range: 0.0 to 1.0
       })
    return response


response = setup_ollama(model,"") #this should reset the model

prompt = """You are an assitant for somoene playing Zork. Respond to each following prompt with what you will need to advance in the game with the
        goal being to survive and collect all the treasures. The exact game input is being sent to you. Your response will be taken
        directly as input to the game. Be careful to only input game commands otherwise the game will get confused"""
response = setup_ollama(model,prompt)
#print (response.message.content)
prompt = """basic commands for zork include 'go north',
 'go south',
 'go west',
 'go east',
 'go northeast',
 'go northwest',
 'go southeast',
 'go southwest',
 'd',
 'u'"""
response = setup_ollama(model,prompt)

prompt = """additional verbs include: ['open OBJ', 'get OBJ', 'set OBJ', 'hit OBJ', 'eat OBJ', 'put OBJ', 'cut OBJ', 'dig OBJ', 'ask OBJ',
 'fix OBJ', 'make OBJ', 'wear OBJ', 'move OBJ', 'kick OBJ', 'kill OBJ', 'find OBJ', 'play OBJ', 'feel OBJ', 'hide OBJ', 'read OBJ',
 'fill OBJ', 'flip OBJ', 'burn OBJ', 'pick OBJ', 'pour OBJ', 'pull OBJ', 'apply OBJ', 'leave OBJ', 'ask OBJ', 'break OBJ', 'enter OBJ',
 'curse OBJ', 'shake OBJ', 'burn OBJ', 'inflate OBJ', 'brandish OBJ', 'donate OBJ', 'squeeze OBJ', 'attach OBJ', 'banish OBJ',
 'enchant OBJ', 'feel OBJ', 'pour OBJ'"""
response = setup_ollama(model,prompt)

play_zork()
