import React from "react";
//import { render } from 'react-dom';
import { createRoot } from 'react-dom/client';

import Header from "./components/Header";
import Todos from "./components/Todos";

function App() {
  return (
    <>
      <Header />
      <Todos />
     
    </>
  )
}

const rootElement = document.getElementById("root")
createRoot(rootElement).render(<App />)
//render(<App />, rootElement)
