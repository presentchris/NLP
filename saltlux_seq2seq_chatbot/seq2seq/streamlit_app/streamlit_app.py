import streamlit as st
import numpy as  np
import time
from seq2seq_SaultluxChatbot_BahdanauAttention import answer


# from seq2seq_SaultluxChatbot_BahdanauAttention import *

# model = seq2seq(vocab_size, EMBEDDING_DIM, UNITS, UNITS, BATCH_SIZE, char2idx[end_index])
# model.compile(loss = loss, optimizer=keras.optimizers.Adam(1e-3))
# model.build([(20, 25), [20, 25]])
# model.load_weights("./data_out/csv_short/seq2seq_kor_BahdanauAttention/weights_attention_pretrained.h5")

st.title("Seq2Seq GRU Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
# := operator to assign the user's input to the prompt variable and checked if it's not None in the same line
if prompt := st.chat_input("What is up?"): # it looks like c#:bool result = Int.TryParse(x, out int result)
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    #################################################################################################
    response = answer(prompt)
    # query = prompt
    # test_index_inputs, _ = enc_processing([query], char2idx) 
    # # print(test_index_inputs)
    # predict_tokens = model.inference(test_index_inputs)
    # # print(predict_tokens)
    # answer = ' '.join([idx2char[str(t)] for t in predict_tokens])
    ##################################################################################################

    time.sleep(1)
    response = f"{response}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

js = f"""
<script>
    function scroll(dummy_var_to_force_repeat_execution){{
        var textAreas = parent.document.querySelectorAll('.stTextArea textarea');
        for (let index = 0; index < textAreas.length; index++) {{
            textAreas[index].style.color = 'red'
            textAreas[index].scrollTop = textAreas[index].scrollHeight;
        }}
    }}
    scroll({len(st.session_state.messages)})
</script>
"""

st.components.v1.html(js)


# with st.chat_message("user"):
#     st.write("Hello 👋")

# with st.chat_message("assistant"):
#     st.write("Hello human")
#     # st.bar_chart(np.random.randn(30, 3))

# prompt = st.chat_input("Say something")
# if prompt:
#     st.write(f"User: {prompt}")
