import { DocsGPTWidget, SearchBar } from 'docsgpt-react';

function App() {
  return (
    <>
      <div>
        <SearchBar
          apiKey=""
          apiHost="http://localhost:7091"
          theme="light"
          placeholder="Search or Ask AI..."
          width="100%"
        />
        <DocsGPTWidget
          apiHost="http://localhost:7091"
          apiKey=""
          avatar = "https://d3dg1063dc54p9.cloudfront.net/cute-docsgpt.png"
          title = "Get AI assistance"
          description = "DocsGPT's AI Chatbot is here to help"
          heroTitle = "Welcome to DocsGPT !"
          heroDescription="This chatbot is built with DocsGPT and utilises GenAI, 
          please review important information using sources."
          theme = "dark"
          buttonIcon = "https://your-icon"
          buttonBg = "#222327"
        />
      </div>
    </>
  )
}

export default App
