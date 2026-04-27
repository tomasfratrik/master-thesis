FROM node:20-slim

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

RUN npm run prepare

EXPOSE 4173

CMD ["sh", "-c", "npm run prepare && npm run dev -- --host 0.0.0.0 --port 4173"]
