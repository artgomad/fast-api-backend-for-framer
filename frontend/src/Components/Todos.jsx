import React, { useEffect, useState } from "react";

const TodosContext = React.createContext({
  todos: [],
  fetchTodos: () => {},
});

function AddTodo() {
  const [item, setItem] = React.useState("");
  const { todos, fetchTodos } = React.useContext(TodosContext);

  const handleInput = (event) => {
    setItem(event.target.value);
  };

  const handleSubmit = (event) => {
    const newTodo = {
      id: todos.length + 1,
      item: item,
    };

    fetch("http://localhost:8000/todo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(newTodo),
    }).then(fetchTodos);
  };

  return (
    <form onSubmit={handleSubmit}>
      <div style={{ display: "flex", alignItems: "center" }}>
        <input
          style={{ marginRight: "0.5rem" }}
          type="text"
          placeholder="Add a todo item"
          aria-label="Add a todo item"
          onChange={handleInput}
        />
        <button type="submit" style={{ padding: "0.5rem" }}>
          Add
        </button>
      </div>
    </form>
  );
}

function UpdateTodo({ item, id }) {
  const [isOpen, setIsOpen] = useState(false);
  const [todo, setTodo] = useState(item);
  const { fetchTodos } = React.useContext(TodosContext);

  const openModal = () => {
    setIsOpen(true);
  };

  const closeModal = () => {
    setIsOpen(false);
  };

  const updateTodo = async () => {
    await fetch(`http://localhost:8000/todo/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item: todo }),
    });
    closeModal();
    await fetchTodos();
  };

  return (
    <>
      <button
        style={{ height: "1.5rem", marginRight: "0.5rem" }}
        onClick={openModal}
      ></button>
      {isOpen && (
        <div>
          <div
            onClick={closeModal}
            style={{
              position: "fixed",
              top: 0,
              right: 0,
              bottom: 0,
              left: 0,
              zIndex: 9998,
            }}
          />
          <div
            style={{
              position: "fixed",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              zIndex: 9999,
            }}
          >
            <div>
              <h2>Update Todo</h2>
              <input
                type="text"
                value={todo}
                onChange={(e) => setTodo(e.target.value)}
              />
              <button onClick={updateTodo}>Update Todo</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function DeleteTodo({ id }) {
  const { fetchTodos } = React.useContext(TodosContext);

  const deleteTodo = async () => {
    await fetch(`http://localhost:8000/todo/${id}`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: { id: id },
    });
    await fetchTodos();
  };

  return (
    <button
      style={{ height: "1.5rem", marginRight: "0.5rem" }}
      onClick={deleteTodo}
    >
      Delete Todo
    </button>
  );
}

function TodoHelper({ item, id, fetchTodos }) {
  return (
    <div
      style={{
        padding: "0.5rem",
        boxShadow: "0 0.125rem 0.25rem rgba(0, 0, 0, 0.075)",
        marginBottom: "0.5rem",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div style={{ marginTop: "0.5rem" }}>{item}</div>
        <div style={{ display: "flex", alignItems: "center" }}>
          <UpdateTodo item={item} id={id} fetchTodos={fetchTodos} />
          <DeleteTodo id={id} fetchTodos={fetchTodos} />
        </div>
      </div>
    </div>
  );
}

export default function Todos() {
  const [todos, setTodos] = useState([]);
  const fetchTodos = async () => {
    const response = await fetch("http://localhost:8000/todo");
    const todos = await response.json();
    setTodos(todos.data);
  };
  useEffect(() => {
    fetchTodos();
  }, []);
  return (
    <TodosContext.Provider value={{ todos, fetchTodos }}>
      <AddTodo />
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        {todos.map((todo) => (
          <TodoHelper item={todo.item} id={todo.id} fetchTodos={fetchTodos} />
        ))}
      </div>
    </TodosContext.Provider>
  );
}
